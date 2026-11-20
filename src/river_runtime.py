"""CPU-only causal event contracts; synthetic time is not measured runtime.

No model, simulator, actuator, worker, or wall-clock timer is started here.
See docs/RUNTIME_CONTRACT.md for the estimator and accounting assumptions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
import heapq
import math
from typing import Any, Callable, Generic, Protocol, TypeVar


State = TypeVar("State")
Vector = tuple[float, ...]


class ContractError(ValueError):
    """An input would violate a causal or accounting contract."""


class BudgetExceeded(ContractError):
    pass


class ResultRejected(ContractError):
    pass


class MotionConstraintViolation(ContractError):
    """A reported physical displacement exceeded its reserved envelope."""


def _time(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ContractError(f"{name} must be finite and nonnegative")
    return value


def _vector(value: Vector) -> Vector:
    value = tuple(float(item) for item in value)
    if not value or not all(math.isfinite(item) for item in value):
        raise ContractError("displacement must be a nonempty finite vector")
    return value


def _norm(value: Vector) -> float:
    return math.sqrt(sum(item * item for item in value))


class ScheduleMode(str, Enum):
    COMPUTE_THEN_MOVE = "compute_then_move"
    MOVE_WITH_LIGHTWEIGHT = "move_with_lightweight"
    COMPUTE_IN_FLIGHT = "compute_in_flight"


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: str
    captured_at: float

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ContractError("frame_id is required")
        object.__setattr__(self, "captured_at", _time(self.captured_at, "captured_at"))


@dataclass(frozen=True)
class InferenceResult:
    result_id: str
    evidence_id: str
    job_id: str
    frame_id: str
    captured_at: float
    compute_started_at: float
    compute_ended_at: float
    delivered_at: float
    evidence: Any

    def __post_init__(self) -> None:
        if not all((self.result_id, self.evidence_id, self.job_id, self.frame_id)):
            raise ContractError("result, evidence, job and frame IDs are required")
        names = ("captured_at", "compute_started_at", "compute_ended_at", "delivered_at")
        stamps = tuple(_time(getattr(self, name), name) for name in names)
        if stamps != tuple(sorted(stamps)):
            raise ContractError("capture <= compute start <= compute end <= delivery required")
        for name, stamp in zip(names, stamps):
            object.__setattr__(self, name, stamp)


@dataclass(frozen=True)
class ActionRequest:
    displacement: Vector
    response_delay: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "displacement", _vector(self.displacement))
        object.__setattr__(self, "response_delay", _time(self.response_delay, "response_delay"))


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    frame_id: str
    captured_at: float
    compute_started_at: float
    compute_bound: float
    deadline: float
    mode: ScheduleMode
    status: str = "running"
    compute_ended_at: float | None = None
    delivered_at: float | None = None
    belief_updated_at: float | None = None
    used_for_action_at: float | None = None


@dataclass(frozen=True)
class ActionRecord:
    command_id: str
    command: Vector
    issued_at: float
    effect_due_at: float
    deadline: float
    evidence_ids: tuple[str, ...]
    is_fallback: bool = False
    effect_at: float | None = None
    realized_displacement: Vector | None = None


@dataclass(frozen=True)
class AuditEvent:
    at: float
    kind: str
    subject: str
    reason: str = ""


@dataclass(frozen=True)
class CostLedger:
    elapsed_task_seconds: float
    committed_compute_seconds: float
    reported_compute_seconds: float
    jobs_submitted: int
    commanded_distance: float
    realized_distance: float
    reserved_motion_distance: float
    deadline_misses: int


class ReplayEstimator(Protocol[State]):
    """All operations must be pure, deterministic, and return fresh states.

    observe receives evidence at its ORIGINAL capture time. Replaying must
    reproduce process noise, association inputs, calibration and camera changes;
    a stochastic adapter must retain draws or use stable per-transition seeds.
    Replaying must never reissue an action or spend a compute budget.
    """

    def propagate(self, state: State, from_at: float, to_at: float) -> State: ...

    def observe(self, state: State, evidence: Any, captured_at: float) -> State: ...

    def camera_motion(self, state: State, displacement: Vector, effect_at: float) -> State: ...


@dataclass(frozen=True)
class KinematicState:
    position: Vector
    velocity: Vector
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        position, velocity = _vector(self.position), _vector(self.velocity)
        if len(position) != len(velocity):
            raise ContractError("position and velocity dimensions differ")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "timestamp", _time(self.timestamp, "timestamp"))


@dataclass(frozen=True)
class PositionEvidence:
    """Toy image-plane position, never a claim of calibrated identity evidence."""

    position: Vector
    velocity: Vector | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector(self.position))
        if self.velocity is not None:
            object.__setattr__(self, "velocity", _vector(self.velocity))
            if len(self.velocity) != len(self.position):
                raise ContractError("evidence dimensions differ")


class ConstantVelocityReplay:
    """Deterministic toy estimator for contract counterexamples only.

    Camera displacement is an instantaneous image-plane subtraction. Real
    integration needs a projection model and actual pose/odometry segments.
    """

    def propagate(self, state: KinematicState, from_at: float, to_at: float) -> KinematicState:
        if state.timestamp != from_at or to_at < from_at:
            raise ContractError("estimator propagation timestamps disagree")
        dt = to_at - from_at
        return replace(state, position=tuple(p + v * dt for p, v in
                                            zip(state.position, state.velocity)), timestamp=to_at)

    def observe(self, state: KinematicState, evidence: PositionEvidence,
                captured_at: float) -> KinematicState:
        if captured_at != state.timestamp or len(evidence.position) != len(state.position):
            raise ContractError("observation must match capture-time state and dimensions")
        return KinematicState(evidence.position, evidence.velocity or state.velocity, captured_at)

    def camera_motion(self, state: KinematicState, displacement: Vector,
                      effect_at: float) -> KinematicState:
        if effect_at != state.timestamp or len(displacement) != len(state.position):
            raise ContractError("motion must match effect-time state and dimensions")
        return replace(state, position=tuple(p - d for p, d in zip(state.position, displacement)))


@dataclass(frozen=True)
class _HistoryEvent:
    at: float
    kind: str
    key: str
    payload: Any


class CausalRuntime(Generic[State]):
    """Deterministic event harness with a conservative resource admission guard.

    `now`, durations, deadlines, results, and motion are supplied, synthetic data.
    Policy callbacks receive only a copy of the current causally reconstructed
    state; queued future results are not inserted into that state.
    """

    def __init__(self, estimator: ReplayEstimator[State], initial_state: State, *,
                 start_at: float = 0.0, task_deadline: float,
                 max_jobs: int, compute_budget_seconds: float,
                 total_motion_cap: float = 3.0, motion_dimensions: int = 2):
        self.estimator = estimator
        self._initial_state = deepcopy(initial_state)
        self.start_at = _time(start_at, "start_at")
        self.now = self.start_at
        self.task_deadline = _time(task_deadline, "task_deadline")
        self.compute_budget_seconds = _time(compute_budget_seconds, "compute_budget_seconds")
        self.total_motion_cap = _time(total_motion_cap, "total_motion_cap")
        if self.task_deadline <= self.start_at:
            raise ContractError("task_deadline must follow start_at")
        if not isinstance(max_jobs, int) or isinstance(max_jobs, bool) or max_jobs < 0:
            raise ContractError("max_jobs must be a nonnegative integer")
        if not isinstance(motion_dimensions, int) or motion_dimensions < 1:
            raise ContractError("motion_dimensions must be positive")
        self.max_jobs, self.motion_dimensions = max_jobs, motion_dimensions
        self._frames: dict[str, CapturedFrame] = {}
        self._jobs: dict[str, JobRecord] = {}
        self._policies: dict[str, Callable[[State], ActionRequest]] = {}
        self._actions: dict[str, ActionRecord] = {}
        self._history: list[_HistoryEvent] = []
        self._evidence_ids: set[str] = set()
        self._evidence_job: dict[str, str] = {}
        self._received_results: set[str] = set()
        self._received_jobs: set[str] = set()
        self._queued_results: set[str] = set()
        self._queued_motions: set[str] = set()
        self._events: list[tuple[float, int, int, str, Any]] = []
        self._sequence = 0
        self._committed_compute = self._reported_compute = 0.0
        self._commanded = self._realized = self._reserved_motion = 0.0
        self._deadline_misses = 0
        self.motion_fault = False
        self.audit: list[AuditEvent] = []
        self._state = self.estimator.propagate(deepcopy(initial_state), self.start_at, self.start_at)
        self.replay_trace: tuple[str, ...] = ()

    @property
    def state(self) -> State:
        return deepcopy(self._state)

    @property
    def jobs(self) -> dict[str, JobRecord]:
        return self._jobs.copy()

    @property
    def actions(self) -> dict[str, ActionRecord]:
        return self._actions.copy()

    @property
    def ledger(self) -> CostLedger:
        return CostLedger(self.now - self.start_at, self._committed_compute,
                          self._reported_compute, len(self._jobs), self._commanded,
                          self._realized, self._reserved_motion, self._deadline_misses)

    @property
    def accepted_evidence_ids(self) -> frozenset[str]:
        return frozenset(self._evidence_ids)

    def register_frame(self, frame: CapturedFrame) -> None:
        if frame.frame_id in self._frames:
            raise ContractError("frame ID already registered")
        if not self.start_at <= frame.captured_at <= self.now:
            raise ContractError("frame capture is future or predates retained history")
        self._frames[frame.frame_id] = frame
        self.audit.append(AuditEvent(self.now, "frame_registered", frame.frame_id))

    def _deadline(self, deadline: float) -> float:
        deadline = _time(deadline, "deadline")
        if not self.now <= deadline <= self.task_deadline:
            raise ContractError("deadline must be fixed within the remaining task horizon")
        return deadline

    def _check_action(self, request: ActionRequest, deadline: float) -> None:
        if len(request.displacement) != self.motion_dimensions:
            raise ContractError("action dimension differs from runtime")
        if self.now + request.response_delay > deadline:
            raise ContractError("action cannot take effect by the fixed deadline")
        distance = _norm(request.displacement)
        if self.motion_fault and distance:
            raise BudgetExceeded("motion fault inhibits nonzero commands")
        if distance and self._realized + self._reserved_motion + distance > self.total_motion_cap + 1e-12:
            raise BudgetExceeded("total motion envelope exhausted")

    def issue_action(self, command_id: str, request: ActionRequest, *, deadline: float,
                     evidence_ids: tuple[str, ...] = (), is_fallback: bool = False) -> ActionRecord:
        deadline = self._deadline(deadline)
        if not command_id or command_id in self._actions:
            raise ContractError("command ID missing or already used")
        if len(set(evidence_ids)) != len(evidence_ids) or not set(evidence_ids) <= self._evidence_ids:
            raise ContractError("action cites duplicate or unavailable evidence")
        self._check_action(request, deadline)
        record = ActionRecord(command_id, request.displacement, self.now,
                              self.now + request.response_delay, deadline,
                              tuple(evidence_ids), is_fallback)
        self._actions[command_id] = record
        for evidence_id in evidence_ids:
            job_id = self._evidence_job.get(evidence_id)
            if job_id is not None and self._jobs[job_id].used_for_action_at is None:
                self._jobs[job_id] = replace(self._jobs[job_id], used_for_action_at=self.now)
        distance = _norm(request.displacement)
        self._commanded += distance
        self._reserved_motion += distance
        self.audit.append(AuditEvent(self.now, "fallback_issued" if is_fallback else
                                     "action_issued", command_id))
        return record

    def _fallback(self, job_id: str, reason: str) -> None:
        command_id = f"{job_id}/fallback"
        if command_id in self._actions:
            return
        self.issue_action(command_id, ActionRequest((0.0,) * self.motion_dimensions),
                          deadline=self.now, is_fallback=True)
        self.audit.append(AuditEvent(self.now, "fallback_reason", job_id, reason))

    def _push(self, at: float, priority: int, kind: str, payload: Any) -> None:
        self._sequence += 1
        heapq.heappush(self._events, (at, priority, self._sequence, kind, payload))

    def submit_job(self, job_id: str, frame_id: str, *, mode: ScheduleMode,
                   compute_bound: float, deadline: float,
                   action_policy: Callable[[State], ActionRequest] | None = None,
                   in_flight_action: ActionRequest | None = None,
                   evidence_ids: tuple[str, ...] = ()) -> JobRecord:
        """Reserve a declared compute upper bound; it is never refunded.

        CTM calls action_policy only after result delivery and causal replay.
        CIF issues in_flight_action now and does not reissue it on completion.
        A timeout leaves an asynchronous job's compute reservation charged.
        """
        mode = ScheduleMode(mode)
        deadline = self._deadline(deadline)
        compute_bound = _time(compute_bound, "compute_bound")
        if not job_id or job_id in self._jobs:
            raise ContractError("job ID missing or already used")
        if frame_id not in self._frames:
            raise ContractError("unknown frame")
        if mode == ScheduleMode.MOVE_WITH_LIGHTWEIGHT:
            raise ContractError("use move_with_lightweight; no extra job is needed")
        if mode == ScheduleMode.COMPUTE_THEN_MOVE:
            if action_policy is None or in_flight_action is not None:
                raise ContractError("compute-then-move requires only an action policy")
        else:
            if action_policy is not None or in_flight_action is None:
                raise ContractError("compute-in-flight requires only an immediate action")
            self._check_action(in_flight_action, deadline)
            if f"{job_id}/in_flight" in self._actions:
                raise ContractError("in-flight command ID already used")
        if not set(evidence_ids) <= self._evidence_ids or len(set(evidence_ids)) != len(evidence_ids):
            raise ContractError("unavailable or duplicate in-flight evidence")
        if len(self._jobs) >= self.max_jobs:
            raise BudgetExceeded("job count exhausted")
        if self._committed_compute + compute_bound > self.compute_budget_seconds + 1e-12:
            raise BudgetExceeded("compute reservation budget exhausted")
        frame = self._frames[frame_id]
        job = JobRecord(job_id, frame_id, frame.captured_at, self.now, compute_bound,
                        deadline, mode)
        self._jobs[job_id] = job
        self._committed_compute += compute_bound
        if action_policy is not None:
            self._policies[job_id] = action_policy
        if in_flight_action is not None:
            self.issue_action(f"{job_id}/in_flight", in_flight_action, deadline=deadline,
                              evidence_ids=evidence_ids)
        self._push(deadline, 2, "deadline", job_id)
        self.audit.append(AuditEvent(self.now, "compute_started", job_id))
        return job

    def accept_lightweight(self, frame_id: str, evidence_id: str, evidence: Any) -> None:
        if frame_id not in self._frames:
            raise ContractError("unknown frame")
        self._insert_evidence(self._frames[frame_id].captured_at, evidence_id, evidence)

    def move_with_lightweight(self, command_id: str, frame_id: str, evidence_id: str,
                              evidence: Any, request: ActionRequest, *, deadline: float) -> ActionRecord:
        deadline = self._deadline(deadline)
        self._check_action(request, deadline)
        if not command_id or command_id in self._actions:
            raise ContractError("command ID missing or already used")
        self.accept_lightweight(frame_id, evidence_id, evidence)
        return self.issue_action(command_id, request, deadline=deadline, evidence_ids=(evidence_id,))

    def _insert_evidence(self, at: float, evidence_id: str, evidence: Any) -> None:
        if not evidence_id or evidence_id in self._evidence_ids:
            raise ResultRejected("evidence source already used or missing")
        if not self.start_at <= at <= self.now:
            raise ResultRejected("future observation or expired history")
        event = _HistoryEvent(at, "observation", evidence_id, deepcopy(evidence))
        candidate = self._history + [event]
        state, trace = self._reconstruct(candidate, self.now)
        self._history, self._state, self.replay_trace = candidate, state, trace
        self._evidence_ids.add(evidence_id)
        self.audit.append(AuditEvent(self.now, "evidence_accepted", evidence_id))

    def _reconstruct(self, history: list[_HistoryEvent], until: float) -> tuple[State, tuple[str, ...]]:
        state, stamp, trace = deepcopy(self._initial_state), self.start_at, []
        # A camera effect at t is included in an observation captured at t.
        # Stable evidence IDs resolve simultaneous observations, not arrival order.
        for event in sorted(history, key=lambda item: (item.at, item.kind == "observation", item.key)):
            if event.at > until:
                raise ContractError("future history entry")
            state = self.estimator.propagate(state, stamp, event.at)
            if event.kind == "observation":
                state = self.estimator.observe(state, deepcopy(event.payload), event.at)
                trace.append(event.key)
            else:
                state = self.estimator.camera_motion(state, event.payload, event.at)
            stamp = event.at
        return self.estimator.propagate(state, stamp, until), tuple(trace)

    def _validate_result(self, result: InferenceResult) -> JobRecord:
        if result.job_id not in self._jobs:
            raise ResultRejected("unknown job")
        job = self._jobs[result.job_id]
        if result.result_id in self._received_results or result.job_id in self._received_jobs:
            raise ResultRejected("result or job completion already received")
        if result.evidence_id in self._evidence_ids:
            raise ResultRejected("evidence source already used")
        if (result.frame_id != job.frame_id or result.captured_at != job.captured_at or
                result.compute_started_at != job.compute_started_at):
            raise ResultRejected("result provenance differs from submitted frame/job")
        if result.delivered_at != self.now:
            raise ResultRejected("result may only be used at its actual delivery event")
        if result.compute_ended_at - result.compute_started_at > job.compute_bound + 1e-12:
            raise ResultRejected("declared compute bound exceeded; external overrun handling required")
        return job

    def deliver_result(self, result: InferenceResult) -> None:
        job = self._validate_result(result)
        # Accounting happens once even when a timed-out result is discarded.
        self._received_results.add(result.result_id)
        self._received_jobs.add(result.job_id)
        self._reported_compute += result.compute_ended_at - result.compute_started_at
        job = replace(job, compute_ended_at=result.compute_ended_at, delivered_at=self.now)
        self._jobs[job.job_id] = job
        if job.status != "running" or self.now > job.deadline:
            raise ResultRejected("fixed decision deadline expired; result discarded")
        self._insert_evidence(result.captured_at, result.evidence_id, result.evidence)
        self._evidence_job[result.evidence_id] = job.job_id
        job = replace(job, status="completed", belief_updated_at=self.now)
        self._jobs[job.job_id] = job
        self.audit.append(AuditEvent(self.now, "result_delivered", result.result_id))
        if job.mode == ScheduleMode.COMPUTE_THEN_MOVE:
            try:
                request = self._policies[job.job_id](self.state)
                self.issue_action(f"{job.job_id}/after_compute", request, deadline=job.deadline,
                                  evidence_ids=(result.evidence_id,))
            except ContractError as error:
                self._deadline_misses += int("deadline" in str(error))
                self._fallback(job.job_id, str(error))
            else:
                self._jobs[job.job_id] = replace(job, used_for_action_at=self.now)

    def schedule_result(self, result: InferenceResult) -> None:
        """Inject a synthetic completion; never executes inference or sleeps."""
        if result.delivered_at < self.now or result.delivered_at > self.task_deadline:
            raise ContractError("synthetic delivery outside remaining task horizon")
        if result.result_id in self._queued_results or result.result_id in self._received_results:
            raise ResultRejected("result already scheduled or received")
        if result.job_id not in self._jobs:
            raise ResultRejected("unknown job")
        self._queued_results.add(result.result_id)
        self._push(result.delivered_at, 1, "result", deepcopy(result))

    def record_motion(self, command_id: str, displacement: Vector, *, effect_at: float) -> None:
        """Record actual displacement; never replace it by the command.

        One effect record per command is the toy harness contract. An adapter
        needing partial responses must split commands into bounded segments.
        Violations retain the actual motion and inhibit further nonzero commands.
        """
        if command_id not in self._actions:
            raise ContractError("unknown command")
        record = self._actions[command_id]
        displacement = _vector(displacement)
        effect_at = _time(effect_at, "effect_at")
        if len(displacement) != self.motion_dimensions or record.effect_at is not None:
            raise ContractError("motion dimensions differ or effect already recorded")
        if not record.issued_at <= effect_at <= self.now:
            raise ContractError("effect is future or precedes command")
        event = _HistoryEvent(effect_at, "motion", command_id, displacement)
        candidate = self._history + [event]
        state, trace = self._reconstruct(candidate, self.now)
        self._history, self._state, self.replay_trace = candidate, state, trace
        self._actions[command_id] = replace(record, effect_at=effect_at,
                                            realized_displacement=displacement)
        actual, reserved = _norm(displacement), _norm(record.command)
        self._realized += actual
        self._reserved_motion = max(0.0, self._reserved_motion - reserved)
        self.audit.append(AuditEvent(self.now, "motion_reported", command_id))
        if effect_at > record.deadline:
            self._deadline_misses += 1
            self.audit.append(AuditEvent(self.now, "effect_deadline_missed", command_id))
        if actual > reserved + 1e-12 or self._realized > self.total_motion_cap + 1e-12:
            self.motion_fault = True
            self.audit.append(AuditEvent(self.now, "motion_constraint_violation", command_id))
            raise MotionConstraintViolation("actual motion exceeded reserved envelope; actual retained")

    def schedule_motion(self, command_id: str, displacement: Vector, *, effect_at: float) -> None:
        """Inject one synthetic realized effect, independent of its command."""
        effect_at = _time(effect_at, "effect_at")
        if not self.now <= effect_at <= self.task_deadline:
            raise ContractError("synthetic motion outside remaining task horizon")
        if command_id not in self._actions:
            raise ContractError("unknown command")
        if command_id in self._queued_motions or self._actions[command_id].effect_at is not None:
            raise ContractError("motion effect already scheduled or recorded")
        self._queued_motions.add(command_id)
        self._push(effect_at, 0, "motion", (command_id, _vector(displacement)))

    def advance_to(self, until: float) -> None:
        until = _time(until, "until")
        if not self.now <= until <= self.task_deadline:
            raise ContractError("clock must be monotone and within task horizon")
        while self._events and self._events[0][0] <= until:
            at, _, _, kind, payload = heapq.heappop(self._events)
            self.now = at
            self._state, self.replay_trace = self._reconstruct(self._history, at)
            if kind == "result":
                try:
                    self.deliver_result(payload)
                except ResultRejected as error:
                    self.audit.append(AuditEvent(self.now, "result_rejected", payload.result_id, str(error)))
            elif kind == "motion":
                self.record_motion(payload[0], payload[1], effect_at=at)
            else:
                job = self._jobs[payload]
                if job.status == "running":
                    self._jobs[payload] = replace(job, status="timed_out")
                    self._deadline_misses += 1
                    self.audit.append(AuditEvent(self.now, "deadline_missed", payload))
                    self._fallback(payload, "fixed result deadline expired")
        self.now = until
        self._state, self.replay_trace = self._reconstruct(self._history, until)
