"""Non-CARLA E03 perception adapters; no model/thread starts on import.

Inject the verified original YOLO, CausalRepresentativeIdentity and E01
WholeView DINO instances. This module fits no model and drives no simulator.
Processed observation ordinal is deliberately separate from source frame IDs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
import threading
import time
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame_id: str
    source_step: int
    capture_sim_s: float
    capture_mono_ns: int
    jpeg: bytes
    width: int
    height: int
    pose: tuple[float, ...]  # x,y,z,pitch,yaw,roll
    intrinsic: tuple[float, ...]  # row-major 3x3
    capture_origin: str = "sensor_measured"

    def __post_init__(self):
        if (not isinstance(self.frame_id, str) or not self.frame_id or
                type(self.source_step) is not int or self.source_step < 0):
            raise ValueError("stable frame ID and nonnegative source step required")
        if type(self.capture_mono_ns) is not int or self.capture_mono_ns < 0:
            raise ValueError("original monotonic acquisition timestamp required")
        if not math.isfinite(self.capture_sim_s) or self.capture_sim_s < 0:
            raise ValueError("nonnegative simulation acquisition time required")
        if not isinstance(self.jpeg, bytes) or not self.jpeg:
            raise ValueError("immutable original JPEG bytes required")
        if any(type(v) is not int or v <= 0 for v in (self.width, self.height)):
            raise ValueError("positive integer image dimensions required")
        pose, intrinsic = tuple(self.pose), tuple(self.intrinsic)
        if len(pose) != 6 or len(intrinsic) != 9 or not np.isfinite(pose + intrinsic).all():
            raise ValueError("finite capture pose and K required")
        if intrinsic[0] <= 0 or intrinsic[4] <= 0 or intrinsic[6:] != (0., 0., 1.):
            raise ValueError("invalid pinhole calibration")
        if not np.allclose([intrinsic[1], intrinsic[3], intrinsic[4], intrinsic[2], intrinsic[5]],
                           [0., 0., intrinsic[0], self.width/2., self.height/2.],
                           rtol=0., atol=1e-9):
            raise ValueError("legacy HSV fov adapter requires zero skew, square pixels and centered K")
        if self.capture_origin not in {"sensor_measured", "archived_replay_ingress"}:
            raise ValueError("explicit sensor/replay capture origin required")
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "intrinsic", intrinsic)

    @property
    def jpeg_sha256(self):
        return hashlib.sha256(self.jpeg).hexdigest()

    def rgb(self):
        from PIL import Image
        image = np.asarray(Image.open(io.BytesIO(self.jpeg)).convert("RGB"))
        if image.shape != (self.height, self.width, 3):
            raise ValueError("JPEG shape differs from capture receipt")
        return image


@dataclass(frozen=True)
class Candidate:
    xyxy: tuple[float, ...]
    confidence: float
    class_id: int
    packed_mask: bytes

    @classmethod
    def from_observed(cls, row, frame):
        if set(row) != {"xyxy", "confidence", "class_id", "mask"}:
            raise ValueError("only original observed candidate fields allowed")
        box, mask = tuple(float(v) for v in row["xyxy"]), np.asarray(row["mask"])
        if (len(box) != 4 or not np.isfinite(box).all() or
                not (0 <= box[0] < box[2] <= frame.width and 0 <= box[1] < box[3] <= frame.height)):
            raise ValueError("invalid original candidate rectangle")
        if mask.shape != (frame.height, frame.width) or not np.isin(mask, [0, 1]).all():
            raise ValueError("candidate mask is not a current binary full-frame mask")
        confidence = float(row["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1 or row["class_id"] != 2:
            raise ValueError("invalid frozen car candidate")
        return cls(box, confidence, 2, np.packbits(mask.astype(np.uint8).ravel()).tobytes())

    def observed(self, frame):
        mask = np.unpackbits(np.frombuffer(self.packed_mask, np.uint8))[:frame.width*frame.height]
        return dict(xyxy=list(self.xyxy), confidence=self.confidence, class_id=self.class_id,
                    mask=mask.reshape(frame.height, frame.width).copy())


@dataclass(frozen=True)
class LightSnapshot:
    frame: FramePacket
    candidates: tuple[Candidate, ...]
    processed_ordinal: int
    skipped_source_steps: tuple[int, ...]
    identity_json: str
    initial_box: tuple[float, ...] | None = None

    @property
    def candidate_sha256(self):
        digest = hashlib.sha256()
        for candidate in self.candidates:
            digest.update(json.dumps([candidate.xyxy, candidate.confidence, candidate.class_id],
                                     separators=(",", ":"), allow_nan=False).encode())
            digest.update(candidate.packed_mask)
        return digest.hexdigest()


class LightAdapter:
    """Actual YOLO.predict + existing frozen causal HSV; no DINO dependency.

    One worker owns this state. Bootstrap source_step=0 is mandatory and must
    be protected by LatestFrameWorker's bootstrap argument, never latest-drop.
    """
    def __init__(self, detector, identity, initial_box):
        self.detector, self.identity = detector, identity
        self.initial_box = tuple(float(v) for v in initial_box)
        if len(self.initial_box) != 4 or not np.isfinite(self.initial_box).all():
            raise ValueError("finite legal first task rectangle required")
        self.ordinal, self.previous_source_step = -1, -1

    def __call__(self, frame: FramePacket):
        if not isinstance(frame, FramePacket) or frame.source_step <= self.previous_source_step:
            raise ValueError("light source steps must be strictly increasing")
        if self.ordinal < 0 and frame.source_step != 0:
            raise ValueError("legal source frame zero bootstrap cannot be skipped")
        rgb = frame.rgb()
        shape, observed = self.detector.predict(frame.jpeg)
        if tuple(shape) != (frame.height, frame.width):
            raise ValueError("detector shape differs from capture")
        candidates = tuple(Candidate.from_observed(row, frame) for row in observed)
        ordinal = self.ordinal + 1
        box = self.initial_box if ordinal == 0 else None
        if box is not None and not (0 <= box[0] < box[2] <= frame.width and
                                    0 <= box[1] < box[3] <= frame.height):
            raise ValueError("legal bootstrap rectangle outside image")
        pose = dict(zip(("x", "y", "z", "pitch", "yaw", "roll"), frame.pose))
        fov = math.degrees(2 * math.atan(frame.width / (2 * frame.intrinsic[0])))
        _, identity = self.identity.update(rgb, [c.observed(frame) for c in candidates],
                                          ordinal, pose, box, fov)
        # ordinal is consumed only by the frozen extractor's sequencing guard.
        # It is never returned as the source sensor step or an intact prefix.
        detail = dict(identity, processed_ordinal=ordinal, source_step=frame.source_step,
                      source_frame_id=frame.frame_id, complete_source_prefix=False)
        result = LightSnapshot(frame, candidates, ordinal,
            tuple(range(self.previous_source_step+1, frame.source_step)),
            json.dumps(detail, sort_keys=True, allow_nan=False), box)
        self.ordinal, self.previous_source_step = ordinal, frame.source_step
        return result


class DeepAdapter:
    """E01 WholeView descriptors on an immutable LightSnapshot; no detector.

    feature_factory must return the SAME 1024-dimensional E01 DINOv3-L16
    WholeView extractor. Native DINOv3-S and foreground DINOv2 are incompatible.
    Loading and lazy first-frame reference extraction occur inside the job and
    therefore remain visible in its measured wall time and call count.
    """
    def __init__(self, feature_factory: Callable[[], Any]):
        self.feature_factory, self.feature = feature_factory, None
        self.reference: LightSnapshot | None = None
        self.positive = self.negatives = None

    def register_reference(self, snapshot: LightSnapshot):
        if (self.reference is not None or snapshot.frame.source_step != 0 or
                snapshot.processed_ordinal != 0 or snapshot.initial_box is None):
            raise ValueError("register exactly one legal light bootstrap reference")
        # Frozen dataclasses/bytes own all images, boxes and masks.
        self.reference = snapshot

    def _extract(self, snapshot, boxes):
        rgb = snapshot.frame.rgb()
        values = self.feature.extract_features(rgb[:, :, ::-1].copy(), np.asarray(boxes, np.float32))
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()  # host transfer waits for actual GPU output
        values = np.asarray(values, np.float64)
        if values.shape != (len(boxes), 1024) or not np.isfinite(values).all():
            raise ValueError("E01 DINO descriptor shape/value mismatch")
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        if (norm <= 1e-12).any():
            raise ValueError("invalid zero-norm descriptor")
        return values / norm

    def __call__(self, snapshot: LightSnapshot):
        if not isinstance(snapshot, LightSnapshot) or self.reference is None:
            raise ValueError("deep requires original light snapshot and registered bootstrap")
        base = dict(frame_id=snapshot.frame.frame_id, source_step=snapshot.frame.source_step,
                    capture_mono_ns=snapshot.frame.capture_mono_ns,
                    capture_sim_s=snapshot.frame.capture_sim_s,
                    capture_pose=snapshot.frame.pose, capture_intrinsic=snapshot.frame.intrinsic,
                    capture_origin=snapshot.frame.capture_origin,
                    jpeg_sha256=snapshot.frame.jpeg_sha256,
                    candidate_sha256=snapshot.candidate_sha256,
                    candidate_count=len(snapshot.candidates), feature_extract_calls=0,
                    scope="E01_descriptors_only_not_new_identity_posterior")
        if not snapshot.candidates:
            return dict(base, status="no_candidates", scores=[])
        loaded = self.feature is None
        if loaded:
            self.feature = self.feature_factory()
        calls = 0
        if self.positive is None:
            box = self.reference.initial_box
            negatives = [c.xyxy for c in self.reference.candidates if
                min(c.xyxy[2], box[2]) <= max(c.xyxy[0], box[0]) or
                min(c.xyxy[3], box[3]) <= max(c.xyxy[1], box[1])]
            refs = self._extract(self.reference, [box] + negatives)
            self.positive, self.negatives = refs[0], refs[1:]
            calls += 1
        values = self._extract(snapshot, [c.xyxy for c in snapshot.candidates])
        calls += 1
        scores = [dict(candidate_index=i, positive_similarity=float(value @ self.positive),
                       known_negative_similarity=float(np.max(value @ self.negatives.T))
                           if len(self.negatives) else None,
                       negative_reference_present=bool(len(self.negatives)),
                       negative_reference_count=len(self.negatives)) for i, value in enumerate(values)]
        return dict(base, status="ok", scores=scores, feature_extract_calls=calls,
                    model_loaded_inside_job=loaded)


@dataclass(frozen=True)
class WorkItem:
    job_id: str
    payload: FramePacket | LightSnapshot
    expires_mono_ns: int

    def __post_init__(self):
        if (not isinstance(self.job_id, str) or not self.job_id or
                not isinstance(self.payload, (FramePacket, LightSnapshot)) or
                type(self.expires_mono_ns) is not int or self.expires_mono_ns < 0):
            raise ValueError("source-bound job and integer monotonic expiry required")

    @property
    def frame(self):
        return self.payload.frame if isinstance(self.payload, LightSnapshot) else self.payload


@dataclass(frozen=True)
class Completion:
    job_id: str
    mode: str
    frame_id: str
    capture_mono_ns: int
    enqueued_mono_ns: int
    compute_started_mono_ns: int
    compute_ended_mono_ns: int
    expires_mono_ns: int
    status: str
    payload: Any = None
    error: str | None = None
    delivered_mono_ns: int | None = None

    @property
    def expired_at_delivery(self):
        return self.delivered_mono_ns is not None and self.delivered_mono_ns > self.expires_mono_ns


class LatestFrameWorker:
    """One running + one replaceable pending request; one completed result.

    poll() records actual controller delivery. No sleeps, artificial durations,
    precomputed completion times, model result fabrication, CARLA or sockets.
    A protected bootstrap executes first. Do not use GPU thread cancellation.
    """
    def __init__(self, mode, processor, *, bootstrap: WorkItem | None = None,
                 max_audit_events=2048):
        if mode not in {"light", "deep"}:
            raise ValueError("light or deep worker required")
        if bootstrap is not None and (mode != "light" or not isinstance(bootstrap, WorkItem) or
                not isinstance(bootstrap.payload, FramePacket) or bootstrap.frame.source_step != 0):
            raise ValueError("protected bootstrap must be light source frame zero")
        self.mode, self.processor = mode, processor
        self._condition = threading.Condition()
        self._pending = self._completed = None
        self._bootstrap, self._bootstrap_enqueue = bootstrap, time.monotonic_ns()
        if bootstrap and bootstrap.frame.capture_mono_ns > self._bootstrap_enqueue:
            raise ValueError("bootstrap acquisition is in the future")
        self._seen = {bootstrap.job_id} if bootstrap else set()
        self._audit, self._max_audit = deque(), max_audit_events
        self._stopping, self._thread, self.audit_overflow = False, None, False

    def _event(self, kind, job_id):
        if len(self._audit) >= self._max_audit:
            self.audit_overflow = self._stopping = True
            return
        self._audit.append(dict(kind=kind, job_id=job_id, mono_ns=time.monotonic_ns()))

    def start(self):
        with self._condition:
            if self._thread is not None:
                raise ValueError("worker already started")
            self._thread = threading.Thread(target=self._run, name="e03-"+self.mode, daemon=True)
            self._thread.start()

    def submit(self, item: WorkItem):
        now = time.monotonic_ns()
        if not isinstance(item, WorkItem) or not isinstance(item.frame, FramePacket):
            raise ValueError("immutable source-bound work item required")
        if ((self.mode == "light") != isinstance(item.payload, FramePacket)):
            raise ValueError("light takes source frames; deep takes original light snapshots")
        if item.frame.capture_mono_ns > now or not item.job_id:
            raise ValueError("future capture or missing job ID")
        with self._condition:
            if self._stopping or item.job_id in self._seen:
                raise ValueError("stopped worker or duplicate job ID")
            self._seen.add(item.job_id)
            if item.expires_mono_ns <= now:
                self._event("expired_before_enqueue", item.job_id)
                return False
            if self._pending:
                self._event("replaced_before_start", self._pending[0].job_id)
            self._pending = (item, now)
            self._event("enqueued", item.job_id)
            self._condition.notify_all()
            return True

    def poll(self):
        with self._condition:
            result, self._completed = self._completed, None
            if result is not None:
                result = replace(result, delivered_mono_ns=time.monotonic_ns())
                self._event("delivered", result.job_id)
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
            return int(self._pending is not None)

    def _run(self):
        while True:
            with self._condition:
                # Backpressure only this worker; world owner keeps ticking.
                while not self._stopping and self._completed is not None:
                    self._condition.wait()
                while not self._stopping and self._bootstrap is None and self._pending is None:
                    self._condition.wait()
                if self._stopping:
                    return
                if self._bootstrap is not None:
                    item, enqueued = self._bootstrap, self._bootstrap_enqueue
                    self._bootstrap = None
                else:
                    item, enqueued = self._pending
                    self._pending = None
                started = time.monotonic_ns()
                self._event("started", item.job_id)
            payload, error = None, None
            status = "expired_before_start"
            if started <= item.expires_mono_ns:
                try:
                    payload, status = self.processor(item.payload), "ok"
                except BaseException as exc:
                    error, status = repr(exc), "error"
            ended = time.monotonic_ns()
            result = Completion(item.job_id, self.mode, item.frame.frame_id,
                item.frame.capture_mono_ns, enqueued, started, ended,
                item.expires_mono_ns, status, payload, error)
            with self._condition:
                self._completed = result
                self._event(status, item.job_id)
                if status == "error":
                    self._stopping = True
                self._condition.notify_all()

    def close(self, timeout=1.):
        with self._condition:
            self._stopping = True
            if self._pending:
                self._event("closed_before_start", self._pending[0].job_id)
                self._pending = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
        return self._thread is None or not self._thread.is_alive()
