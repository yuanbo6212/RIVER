"""Bounded, in-order frame intake: drop-in for LatestFrameWorker.

`river_e03_perception.LatestFrameWorker` keeps exactly one replaceable pending
request. Under load a frame can be replaced before the worker starts it, and
the frozen schedule's detection for that step never happens. This class keeps
the same public surface but replaces the single slot with a bounded FIFO:

  * `submit` appends while `len(queue) < capacity`; otherwise it REJECTS the
    item and records why. It never silently discards an already-accepted item.
  * `_run` takes the oldest item, so source-frame order is preserved.
  * Delivery keeps the original one-result-at-a-time backpressure, so the
    world owner still never waits for a model.
  * `rejected` counts every rejection with a reason; `intake_log` keeps a
    bounded audit of accept/reject/expire decisions for the run receipt.

The bootstrap slot, expiry rule, error-stop behaviour and `poll` semantics are
unchanged from the frozen class, so existing callers need no edits.

No CARLA, no model, no sockets. Pure CPU contract.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import replace

from river_e03_perception import Completion, FramePacket, WorkItem


class BoundedFrameWorker:
    """Bounded FIFO intake with explicit, accounted rejection."""

    def __init__(self, mode, processor, *, bootstrap=None, capacity=8,
                 max_audit_events=2048):
        if mode not in {'light', 'deep'}:
            raise ValueError('light or deep worker required')
        if capacity < 1:
            raise ValueError('capacity must be at least one')
        if bootstrap is not None:
            if (mode != 'light' or not isinstance(bootstrap, WorkItem) or
                    not isinstance(bootstrap.payload, FramePacket) or
                    bootstrap.frame.source_step != 0):
                raise ValueError('protected bootstrap must be light source frame zero')
        self.mode, self.processor = mode, processor
        self.capacity = int(capacity)
        self._pending = deque()
        self._completed = None
        self._bootstrap, self._bootstrap_enqueue = bootstrap, time.monotonic_ns()
        if bootstrap and bootstrap.frame.capture_mono_ns > self._bootstrap_enqueue:
            raise ValueError('bootstrap acquisition is in the future')
        self._seen = {bootstrap.job_id} if bootstrap else set()
        self._audit, self._max_audit = deque(), max_audit_events
        self.intake_log = deque(maxlen=max_audit_events)
        self.rejected = {}
        self.audit_overflow = False
        self._stopping = False
        self._thread = None
        self._condition = __import__('threading').Condition()

    # --- bookkeeping -------------------------------------------------------
    def _event(self, kind, job_id):
        if len(self._audit) >= self._max_audit:
            self.audit_overflow = self._stopping = True
            return
        self._audit.append(dict(kind=kind, job_id=job_id, mono_ns=time.monotonic_ns()))

    def _record(self, decision, item, reason=None):
        self.intake_log.append(dict(
            decision=decision, job_id=item.job_id if item else None,
            source_step=item.frame.source_step if item else None,
            reason=reason, queue_depth=len(self._pending),
            mono_ns=time.monotonic_ns()))
        if decision == 'rejected':
            self.rejected[reason] = self.rejected.get(reason, 0) + 1

    # --- lifecycle ---------------------------------------------------------
    def start(self):
        import threading
        with self._condition:
            if self._thread is not None:
                raise ValueError('worker already started')
            self._thread = threading.Thread(target=self._run, name='e03-bounded-' + self.mode,
                                            daemon=True)
            self._thread.start()

    def submit(self, item):
        now = time.monotonic_ns()
        if not isinstance(item, WorkItem) or not isinstance(item.frame, FramePacket):
            raise ValueError('immutable source-bound work item required')
        if (self.mode == 'light') != isinstance(item.payload, FramePacket):
            raise ValueError('light takes source frames; deep takes original light snapshots')
        if item.frame.capture_mono_ns > now or not item.job_id:
            raise ValueError('future capture or missing job ID')
        with self._condition:
            if self._stopping or item.job_id in self._seen:
                raise ValueError('stopped worker or duplicate job ID')
            self._seen.add(item.job_id)
            if item.expires_mono_ns <= now:
                self._event('expired_before_enqueue', item.job_id)
                self._record('rejected', item, 'expired_before_enqueue')
                return False
            if len(self._pending) >= self.capacity:
                # Explicit refusal. Nothing already accepted is discarded.
                self._event('queue_full_rejected', item.job_id)
                self._record('rejected', item, 'queue_full')
                return False
            self._pending.append((item, now))
            self._event('enqueued', item.job_id)
            self._record('accepted', item)
            self._condition.notify_all()
            return True

    def poll(self):
        with self._condition:
            result, self._completed = self._completed, None
            if result is not None:
                result = replace(result, delivered_mono_ns=time.monotonic_ns())
                self._event('delivered', result.job_id)
            self._condition.notify_all()
            return result

    def drain_audit(self):
        with self._condition:
            events = list(self._audit)
            self._audit.clear()
            return events

    @property
    def pending_count(self):
        with self._condition:
            return len(self._pending)

    @property
    def queue_depth(self):
        with self._condition:
            return len(self._pending)

    def _run(self):
        while True:
            with self._condition:
                while not self._stopping and self._completed is not None:
                    self._condition.wait()
                while (not self._stopping and self._bootstrap is None and
                       not self._pending):
                    self._condition.wait()
                if self._stopping:
                    return
                if self._bootstrap is not None:
                    item, enqueued = self._bootstrap, self._bootstrap_enqueue
                    self._bootstrap = None
                else:
                    item, enqueued = self._pending.popleft()
                started = time.monotonic_ns()
                self._event('started', item.job_id)
            payload, error = None, None
            status = 'expired_before_start'
            if started <= item.expires_mono_ns:
                try:
                    payload, status = self.processor(item.payload), 'ok'
                except BaseException as exc:
                    error, status = repr(exc), 'error'
            ended = time.monotonic_ns()
            result = Completion(item.job_id, self.mode, item.frame.frame_id,
                                item.frame.capture_mono_ns, enqueued, started, ended,
                                item.expires_mono_ns, status, payload, error)
            with self._condition:
                self._completed = result
                self._event(status, item.job_id)
                if status == 'error':
                    self._stopping = True
                self._condition.notify_all()

    def close(self, timeout=1.):
        with self._condition:
            self._stopping = True
            while self._pending:
                item, _ = self._pending.popleft()
                self._event('closed_before_start', item.job_id)
                self._record('rejected', item, 'closed_before_start')
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
        return self._thread is None or not self._thread.is_alive()
