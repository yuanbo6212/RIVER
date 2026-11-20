"""Source-bound E03 job admission and actual completion accounting.

No models, clocks, threads, simulator or estimator are started. Nanosecond
stamps and optional GPU measurements come from the actual execution adapter.
CUDA stream elapsed time and GPU resource seconds are separate quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from river_e03_perception import Completion, FramePacket, LightSnapshot, WorkItem
from river_runtime import ScheduleMode


class AccountingError(ValueError):
    pass


class AdmissionDenied(AccountingError):
    pass


def _ns(value, name):
    if type(value) is not int or value < 0:
        raise AccountingError(name + " requires nonnegative integer monotonic nanoseconds")
    return value


def _seconds(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise AccountingError(name + " requires finite nonnegative seconds")
    return float(value)


def canonical_schedule(value):
    """Only the existing planner/runtime spelling difference is an alias."""
    aliases = {"move_lightweight": "move_with_lightweight"}
    try:
        return ScheduleMode(aliases.get(value, value))
    except (ValueError, TypeError) as exc:
        raise AccountingError("unknown schedule mode") from exc


@dataclass(frozen=True)
class GpuMeasurement:
    cuda_stream_elapsed_s: float | None = None
    gpu_resource_s: float | None = None
    cuda_provenance: str | None = None
    resource_provenance: str | None = None

    def __post_init__(self):
        for name, source in (("cuda_stream_elapsed_s", "cuda_provenance"),
                             ("gpu_resource_s", "resource_provenance")):
            value, provenance = getattr(self, name), getattr(self, source)
            if value is None:
                if provenance is not None:
                    raise AccountingError("unknown measurement cannot have fabricated provenance")
            else:
                _seconds(value, name)
                if not isinstance(provenance, str) or not provenance.strip():
                    raise AccountingError("measured cost needs explicit provenance")


@dataclass(frozen=True)
class SourceBinding:
    frame_id: str
    source_step: int
    capture_mono_ns: int
    capture_sim_s: float
    capture_pose: tuple
    capture_intrinsic: tuple
    capture_origin: str
    jpeg_sha256: str
    candidate_sha256: str | None
    candidate_count: int | None

    @classmethod
    def from_payload(cls, payload):
        frame = payload.frame if isinstance(payload, LightSnapshot) else payload
        if not isinstance(frame, FramePacket):
            raise AccountingError("original immutable source frame required")
        return cls(frame.frame_id, frame.source_step, frame.capture_mono_ns,
                   frame.capture_sim_s, frame.pose, frame.intrinsic, frame.capture_origin,
                   frame.jpeg_sha256,
                   payload.candidate_sha256 if isinstance(payload, LightSnapshot) else None,
                   len(payload.candidates) if isinstance(payload, LightSnapshot) else None)


def _payload_record(payload):
    if isinstance(payload, LightSnapshot):
        return dict(binding=SourceBinding.from_payload(payload).__dict__,
                    processed_ordinal=payload.processed_ordinal,
                    skipped_source_steps=payload.skipped_source_steps,
                    identity=json.loads(payload.identity_json), initial_box=payload.initial_box)
    return payload


def _digest(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise AccountingError("completion receipt must have finite JSON-compatible content") from exc
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SettledJob:
    job_id: str
    mode: str
    schedule: ScheduleMode
    binding: SourceBinding
    status: str
    admitted_mono_ns: int
    enqueued_mono_ns: int
    compute_started_mono_ns: int
    compute_ended_mono_ns: int
    delivered_mono_ns: int
    expires_mono_ns: int
    queue_wall_s: float
    compute_wall_s: float
    capture_to_delivery_wall_s: float
    measurements: GpuMeasurement
    allow_capture_time_update: bool
    reservation_overrun: bool
    budget_overrun: bool
    measurement_pending: bool
    receipt_sha256: str


@dataclass
class _Admission:
    job_id: str
    mode: str
    schedule: ScheduleMode
    binding: SourceBinding
    admitted_mono_ns: int
    expires_mono_ns: int
    reserved_budget_s: float
    marked_start_ns: int | None = None
    decision_expired: bool = False
    settled: SettledJob | None = None


class JobAccounting:
    """Conservative per-task optional-deep budget; common-light costs still logged.

    A missing measurement retains its reservation as an unresolved liability.
    Reaching an expiry never cancels a running GPU call or refunds its cost.
    The hard deep-job limit counts admitted requests, including failed jobs.
    """
    def __init__(self, *, max_deep_jobs=1, max_deep_budget_s,
                 budget_metric="cuda_stream_elapsed_s"):
        if type(max_deep_jobs) is not int or max_deep_jobs < 0:
            raise AccountingError("nonnegative integer deep-job limit required")
        if budget_metric not in {"cuda_stream_elapsed_s", "gpu_resource_s"}:
            raise AccountingError("explicit CUDA-stream or GPU-resource budget unit required")
        self.max_deep_jobs = max_deep_jobs
        self.max_deep_budget_s = _seconds(max_deep_budget_s, "max deep budget")
        self.budget_metric = budget_metric
        self._jobs = {}
        self.audit = []

    def _liability(self, job):
        if job.mode != "deep":
            return 0.
        if job.settled is not None:
            measured = getattr(job.settled.measurements, self.budget_metric)
            if measured is not None:
                return measured
        return job.reserved_budget_s

    def admit(self, item: WorkItem, *, mode, schedule, admitted_mono_ns,
              reserved_budget_s=0.):
        at = _ns(admitted_mono_ns, "admission")
        if not isinstance(item, WorkItem) or mode not in {"light", "deep"}:
            raise AccountingError("source-bound work item and actual perception mode required")
        if (mode == "light") != isinstance(item.payload, FramePacket):
            raise AccountingError("light takes FramePacket; deep takes original LightSnapshot")
        if item.job_id in self._jobs:
            raise AccountingError("duplicate job admission")
        binding = SourceBinding.from_payload(item.payload)
        if binding.capture_mono_ns > at or item.expires_mono_ns <= at:
            raise AdmissionDenied("future capture or expired request")
        reserve = _seconds(reserved_budget_s, "reserved budget")
        if mode == "light" and reserve != 0:
            raise AccountingError("optional-deep reservation cannot be charged to common light")
        if mode == "deep":
            if reserve <= 0:
                raise AdmissionDenied("deep job requires a positive conservative budget reservation")
            deep_jobs = [j for j in self._jobs.values() if j.mode == "deep"]
            if len(deep_jobs) >= self.max_deep_jobs:
                raise AdmissionDenied("hard per-task deep-job limit reached")
            if any(j.settled and (j.settled.reservation_overrun or j.settled.budget_overrun) for j in deep_jobs):
                raise AdmissionDenied("actual compute budget overrun; further optional jobs denied")
            if any(j.settled and j.settled.measurement_pending for j in deep_jobs):
                raise AdmissionDenied("actual optional compute cost unresolved")
            if sum(self._liability(j) for j in deep_jobs) + reserve > self.max_deep_budget_s + 1e-12:
                raise AdmissionDenied("known cost plus outstanding reservation exceeds budget")
        job = _Admission(item.job_id, mode, canonical_schedule(schedule), binding, at,
                         item.expires_mono_ns, reserve)
        self._jobs[item.job_id] = job
        self.audit.append(dict(kind="admitted_not_started", job_id=item.job_id,
                               mono_ns=at, reserved_budget_s=reserve, budget_metric=self.budget_metric))
        return binding

    def mark_started(self, job_id, actual_start_mono_ns):
        job = self._jobs[job_id]
        at = _ns(actual_start_mono_ns, "compute start")
        if at < job.admitted_mono_ns or job.settled is not None:
            raise AccountingError("compute start precedes admission or follows settlement")
        if job.marked_start_ns is not None and at != job.marked_start_ns:
            raise AccountingError("conflicting actual compute start")
        job.marked_start_ns = at

    def expire_decision(self, job_id, now_mono_ns):
        job = self._jobs[job_id]
        at = _ns(now_mono_ns, "decision expiry observation")
        if at <= job.expires_mono_ns:
            raise AccountingError("decision has not passed its declared expiry")
        job.decision_expired = True
        self.audit.append(dict(kind="decision_expired_no_compute_refund", job_id=job_id, mono_ns=at))

    def settle(self, completion: Completion, measurements: GpuMeasurement | None = None):
        if not isinstance(completion, Completion) or completion.job_id not in self._jobs:
            raise AccountingError("completion has no admitted request")
        job = self._jobs[completion.job_id]
        measurement = measurements if measurements is not None else GpuMeasurement()
        if not isinstance(measurement, GpuMeasurement):
            raise AccountingError("typed optional GPU measurements required")
        times = [completion.capture_mono_ns, job.admitted_mono_ns, completion.enqueued_mono_ns,
                 completion.compute_started_mono_ns, completion.compute_ended_mono_ns,
                 completion.delivered_mono_ns]
        for i, value in enumerate(times):
            _ns(value, "completion timestamp " + str(i))
        if times != sorted(times):
            raise AccountingError("capture <= admission <= enqueue <= start <= end <= delivery required")
        if (completion.mode != job.mode or completion.frame_id != job.binding.frame_id or
                completion.capture_mono_ns != job.binding.capture_mono_ns or
                completion.expires_mono_ns != job.expires_mono_ns):
            raise AccountingError("completion source/mode/deadline differs from admitted request")
        if job.marked_start_ns is not None and job.marked_start_ns != completion.compute_started_mono_ns:
            raise AccountingError("completion actual start differs from recorded worker start")
        if completion.status not in {"ok", "error", "expired_before_start"}:
            raise AccountingError("unknown completion status")
        if completion.status == "ok":
            if job.mode == "light":
                if not isinstance(completion.payload, LightSnapshot):
                    raise AccountingError("light completion requires original source snapshot")
                if SourceBinding.from_payload(completion.payload) != job.binding:
                    # The light candidate hash was not known at admission.
                    source = SourceBinding.from_payload(completion.payload)
                    if source.__dict__ | {"candidate_sha256": None, "candidate_count": None} != job.binding.__dict__:
                        raise AccountingError("light completion rebound to another source frame")
            else:
                payload = completion.payload
                if not isinstance(payload, dict):
                    raise AccountingError("deep completion requires source-bound feature record")
                for key, expected in job.binding.__dict__.items():
                    if key not in payload or _digest(payload[key]) != _digest(expected):
                        raise AccountingError("deep completion source binding differs: " + key)
        if completion.status == "expired_before_start":
            if completion.compute_started_mono_ns <= completion.expires_mono_ns:
                raise AccountingError("unstarted expiry contradicts worker timestamps")
            if completion.payload is not None:
                raise AccountingError("unstarted job cannot contain computed evidence")
            # The worker status proves no processor was called; clock overhead
            # remains compute_wall_s, while optional GPU execution was zero.
            zero = GpuMeasurement(0., 0., "worker_verified_no_processor_call", "worker_verified_no_processor_call")
            if measurements is not None and measurement != zero:
                raise AccountingError("unstarted job contradicts supplied GPU charge")
            measurement = zero
        receipt = _digest(dict(completion={**completion.__dict__, "payload": _payload_record(completion.payload)},
                               measurement=measurement.__dict__))
        if job.settled is not None:
            if job.settled.receipt_sha256 != receipt:
                raise AccountingError("conflicting duplicate completion; original settlement retained")
            return job.settled
        measured_budget = getattr(measurement, self.budget_metric)
        overrun = job.mode == "deep" and measured_budget is not None and measured_budget > job.reserved_budget_s + 1e-12
        actual_liability = sum(self._liability(j) for j in self._jobs.values() if j is not job)
        actual_liability += (measured_budget if measured_budget is not None else job.reserved_budget_s) if job.mode == "deep" else 0.
        task_overrun = actual_liability > self.max_deep_budget_s + 1e-12
        result = SettledJob(job.job_id, job.mode, job.schedule, job.binding, completion.status,
            job.admitted_mono_ns, completion.enqueued_mono_ns, completion.compute_started_mono_ns,
            completion.compute_ended_mono_ns, completion.delivered_mono_ns, completion.expires_mono_ns,
            (times[3]-times[2])/1e9, (times[4]-times[3])/1e9, (times[5]-times[0])/1e9,
            measurement, completion.status == "ok" and not job.decision_expired and
                completion.delivered_mono_ns <= completion.expires_mono_ns,
            overrun, task_overrun, job.mode == "deep" and measured_budget is None, receipt)
        job.settled = result
        self.audit.append(dict(kind="settled", job_id=job.job_id, receipt_sha256=receipt,
                               eligible_capture_time_update=result.allow_capture_time_update,
                               reservation_overrun=overrun, budget_overrun=task_overrun,
                               measurement_pending=result.measurement_pending))
        return result

    def summary(self):
        jobs = list(self._jobs.values())
        settled = [j.settled for j in jobs if j.settled is not None]
        return dict(budget_metric=self.budget_metric, max_deep_budget_s=self.max_deep_budget_s,
            deep_jobs_admitted=sum(j.mode == "deep" for j in jobs), max_deep_jobs=self.max_deep_jobs,
            deep_budget_liability_s=sum(self._liability(j) for j in jobs),
            known_cuda_stream_elapsed_s=sum(s.measurements.cuda_stream_elapsed_s for s in settled
                                            if s.measurements.cuda_stream_elapsed_s is not None),
            cuda_unknown_job_ids=[s.job_id for s in settled if s.measurements.cuda_stream_elapsed_s is None],
            known_gpu_resource_s=sum(s.measurements.gpu_resource_s for s in settled
                                     if s.measurements.gpu_resource_s is not None),
            gpu_resource_unknown_job_ids=[s.job_id for s in settled if s.measurements.gpu_resource_s is None],
            settled_jobs=len(settled), pending_jobs=[j.job_id for j in jobs if j.settled is None],
            reservation_overrun_job_ids=[s.job_id for s in settled if s.reservation_overrun],
            budget_overrun_job_ids=[s.job_id for s in settled if s.budget_overrun],
            total_job_compute_wall_s=sum(s.compute_wall_s for s in settled),
            cost_scope="optional deep budget; common light measurements reported separately from resource seconds")
