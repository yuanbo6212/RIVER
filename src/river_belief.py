"""Finite-particle single-target belief core for the RIVER development track.

This module consumes *calibrated densities*, never raw HSV/cosine similarity.
It contains no detector, calibration fit, simulator access, or ground truth.
The model and its limits are specified in docs/BELIEF_THEORY_REVIEW.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np


class ImpossibleObservation(ValueError):
    """The supplied model gives this candidate set zero prior predictive mass."""


def _readonly(value: Any) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _finite_nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _log_nonnegative(value: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(value)


def _logsumexp(value: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    maximum = np.max(value, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        shifted = np.where(np.isfinite(maximum), value - maximum, -np.inf)
        result = maximum + np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


@dataclass(frozen=True)
class ParticleBelief:
    """Normalized weighted point samples [position..., velocity...] at one time.

    Nonnegative input weights are normalized; a zero-mass prior is rejected.
    States are in one consistent Euclidean coordinate frame. Camera/world
    projection and visibility must be supplied by a separate causal model.
    """

    states: np.ndarray
    weights: np.ndarray
    timestamp: float

    def __post_init__(self) -> None:
        states = _readonly(self.states)
        weights = np.array(self.weights, dtype=np.float64, copy=True)
        if states.ndim != 2 or states.shape[0] == 0:
            raise ValueError("states must be a nonempty N x (2*d) array")
        if states.shape[1] < 2 or states.shape[1] % 2:
            raise ValueError("states must contain position then velocity of equal size")
        if not np.all(np.isfinite(states)):
            raise ValueError("states must be finite")
        if weights.shape != (states.shape[0],):
            raise ValueError("weights must have one entry per particle")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be finite and nonnegative")
        scale = weights.max()
        if scale <= 0:
            raise ValueError("prior must have positive total mass")
        weights /= scale
        weights /= weights.sum()
        weights.setflags(write=False)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "timestamp", _finite_nonnegative(self.timestamp, "timestamp"))

    @property
    def dimension(self) -> int:
        return self.states.shape[1] // 2

    def weighted_moments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mean and population covariance of the finite weighted distribution."""
        mean = self.weights @ self.states
        centered = self.states - mean
        covariance = centered.T @ (self.weights[:, None] * centered)
        return mean, covariance

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.dot(self.weights, self.weights))


def cv_transition_and_noise(
    dimension: int, dt: float, acceleration_spectral_density: Any = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Constant velocity with continuous white acceleration (units length²/time³).

    Q = [[q*dt³/3, q*dt²/2], [q*dt²/2, q*dt]], independently per axis.
    This is not a per-frame acceleration variance. The distinction is essential
    when computation or occlusion changes the elapsed time.
    """
    if not isinstance(dimension, (int, np.integer)) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    dt = _finite_nonnegative(dt, "dt")
    q = np.asarray(acceleration_spectral_density, dtype=np.float64)
    if q.ndim == 0:
        q = np.full(dimension, float(q))
    if q.shape != (dimension,) or not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("acceleration spectral density must be nonnegative scalar or d-vector")
    elapsed = np.float64(dt)
    with np.errstate(over="ignore", invalid="ignore"):
        transition = np.eye(2 * dimension)
        transition[:dimension, dimension:] = dt * np.eye(dimension)
        velocity_variance = q * elapsed
        cross_covariance = velocity_variance * (elapsed / 2)
        position_variance = (velocity_variance * (elapsed / 3)) * elapsed
        noise = np.block([
            [np.diag(position_variance), np.diag(cross_covariance)],
            [np.diag(cross_covariance), np.diag(velocity_variance)],
        ])
    if not np.all(np.isfinite(transition)) or not np.all(np.isfinite(noise)):
        raise ValueError("dt or acceleration density overflows the motion covariance")
    return transition, noise


def predict_cv(
    belief: ParticleBelief,
    dt: float,
    acceleration_spectral_density: Any = 0.0,
    *,
    rng: Optional[np.random.Generator] = None,
) -> ParticleBelief:
    """Propagate points and inject dt-aware motion uncertainty without reweighting.

    Nonzero process noise requires an explicit Generator. To replay delayed
    evidence, retain the seed/noise per transition; advancing a shared generator
    again would silently change the history. q=0 needs no generator.
    """
    dt = _finite_nonnegative(dt, "dt")
    transition, noise = cv_transition_and_noise(belief.dimension, dt, acceleration_spectral_density)
    with np.errstate(over="ignore", invalid="ignore"):
        states = belief.states @ transition.T
    if np.any(noise):
        if not isinstance(rng, np.random.Generator):
            raise ValueError("nonzero process noise requires an explicit numpy Generator")
        # Correlated integrated-Wiener increments avoid factoring an almost
        # singular covariance for very small dt. The position residual variance
        # is q*dt^3/12, exactly one quarter of the position marginal variance.
        for axis in range(belief.dimension):
            velocity_axis = axis + belief.dimension
            normals = rng.standard_normal((states.shape[0], 2))
            velocity_increment = np.sqrt(noise[velocity_axis, velocity_axis]) * normals[:, 0]
            position_increment = (
                (dt / 2) * velocity_increment
                + (np.sqrt(noise[axis, axis]) / 2) * normals[:, 1]
            )
            states[:, axis] += position_increment
            states[:, velocity_axis] += velocity_increment
    timestamp = belief.timestamp + dt
    return ParticleBelief(states, belief.weights, timestamp)


@dataclass(frozen=True)
class CalibratedEvidence:
    """Candidate-set likelihood inputs evaluated at the belief's capture time.

    target_density[j,i] = g(z_i|x_j), clutter_intensity[i] = kappa(z_i),
    p_detection[j] = probability the target generates a retained candidate.
    Clutter intensity is a positive density times an expected clutter count,
    not a normalized class probability. Target density may be zero off support.

    Provenance declarations are mandatory but cannot prove calibration; real
    adapters need held-out development validation. Synthetic inputs must be
    explicitly marked synthetic_density/synthetic, never passed off as fitted.
    """

    p_detection: np.ndarray
    target_density: np.ndarray
    clutter_intensity: np.ndarray
    source_timestamp: float
    calibration_id: str
    calibration_split: str
    density_space: str
    evidence_kind: str = "calibrated_density"
    candidate_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        target = _readonly(self.target_density)
        clutter = _readonly(self.clutter_intensity)
        detection = np.asarray(self.p_detection, dtype=np.float64)
        if target.ndim != 2 or target.shape[0] == 0:
            raise ValueError("target_density must be N x M with N > 0, including M=0")
        if detection.ndim == 0:
            detection = np.full(target.shape[0], float(detection))
        detection = _readonly(detection)
        if detection.shape != (target.shape[0],):
            raise ValueError("p_detection must be scalar or have one value per particle")
        if not np.all(np.isfinite(detection)) or np.any((detection < 0) | (detection > 1)):
            raise ValueError("p_detection must lie in [0,1] and be finite")
        if not np.all(np.isfinite(target)) or np.any(target < 0):
            raise ValueError("target densities must be finite and nonnegative")
        if clutter.shape != (target.shape[1],):
            raise ValueError("clutter_intensity must have one value per candidate")
        if not np.all(np.isfinite(clutter)) or np.any(clutter <= 0):
            raise ValueError("clutter intensities must be finite and strictly positive")
        for name in ("calibration_id", "density_space"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        allowed = {("calibrated_density", "development"), ("synthetic_density", "synthetic")}
        if (self.evidence_kind, self.calibration_split) not in allowed:
            raise ValueError("use development-calibrated densities or explicitly synthetic densities; no test fitting")
        ids = tuple(self.candidate_ids)
        if not ids and target.shape[1]:
            ids = tuple(str(i) for i in range(target.shape[1]))
        if len(ids) != target.shape[1] or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("candidate_ids must contain one nonempty opaque string per candidate")
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_ids must be unique within this frame")
        object.__setattr__(self, "target_density", target)
        object.__setattr__(self, "clutter_intensity", clutter)
        object.__setattr__(self, "p_detection", detection)
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "source_timestamp", _finite_nonnegative(self.source_timestamp, "source_timestamp"))

    @classmethod
    def from_policy_payload(cls, payload: Mapping[str, Any]) -> "CalibratedEvidence":
        """Strict causal boundary: no arbitrary metadata or ground-truth fields.

        A producer must derive these numeric arrays from causal observations and
        frozen development calibration. The whitelist does not certify origin.
        """
        allowed = set(cls.__dataclass_fields__)
        unexpected = set(payload) - allowed
        if unexpected:
            raise ValueError(f"unsupported policy evidence fields: {sorted(unexpected)}")
        return cls(**dict(payload))


@dataclass(frozen=True)
class BeliefUpdate:
    posterior: ParticleBelief
    joint_association: np.ndarray
    association_probability: np.ndarray
    log_state_likelihood_ratio: np.ndarray
    log_evidence_ratio: float
    candidate_ids: Tuple[str, ...]

    @property
    def null_probability(self) -> float:
        return float(self.association_probability[0])

    @property
    def candidate_probability(self) -> np.ndarray:
        return self.association_probability[1:]


def update_belief(belief: ParticleBelief, evidence: CalibratedEvidence) -> BeliefUpdate:
    """Normalize one mutually exclusive null-or-single-candidate association.

    For each particle j, use L_j = 1-pD_j + pD_j sum_i(g_ji/kappa_i).
    There is no second negative penalty for incompatible candidates. A=0 is a
    missed target detection, not target nonexistence (existence is fixed to one).
    Log arithmetic avoids forming overflowing density ratios.
    """
    if evidence.target_density.shape[0] != belief.states.shape[0]:
        raise ValueError("evidence and prior must have the same particle count")
    if not np.isclose(evidence.source_timestamp, belief.timestamp, atol=1e-9, rtol=0):
        raise ValueError("update must occur at capture time; replay delayed observations before propagating")
    with np.errstate(divide="ignore"):
        null = np.log1p(-evidence.p_detection)[:, None]
    candidate = (
        _log_nonnegative(evidence.p_detection)[:, None]
        + _log_nonnegative(evidence.target_density)
        - np.log(evidence.clutter_intensity)[None, :]
    )
    log_terms = np.concatenate((null, candidate), axis=1)
    log_joint = _log_nonnegative(belief.weights)[:, None] + log_terms
    log_normalizer = float(_logsumexp(log_joint))
    if not np.isfinite(log_normalizer):
        raise ImpossibleObservation("candidate set has zero predictive mass; check p_detection/support, do not invent a posterior")
    joint = np.exp(log_joint - log_normalizer)
    joint /= joint.sum()
    posterior = ParticleBelief(belief.states, joint.sum(axis=1), belief.timestamp)
    return BeliefUpdate(
        posterior=posterior,
        joint_association=_readonly(joint),
        association_probability=_readonly(joint.sum(axis=0)),
        log_state_likelihood_ratio=_readonly(_logsumexp(log_terms, axis=1)),
        log_evidence_ratio=log_normalizer,
        candidate_ids=evidence.candidate_ids,
    )


@dataclass(frozen=True)
class IdentityRisk:
    decision: str
    accepted_candidate: Optional[int]
    risk: float
    accept_losses: np.ndarray
    unknown_loss: float


def instantaneous_identity_risk(
    association_probability: np.ndarray, wrong_cost: float, unknown_cost: float
) -> IdentityRisk:
    """Bayes action for one frame; index 0 is null, candidate indices start at 0.

    Ties favor unknown. This stage surrogate is not a finite-horizon recovery
    objective and cannot establish terminal 10-frame retention or wrong-run risk.
    """
    probability = np.asarray(association_probability, dtype=np.float64)
    if probability.ndim != 1 or probability.size < 1:
        raise ValueError("association_probability must contain null plus zero or more candidates")
    if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("association probabilities must be finite and in [0,1]")
    if not np.isclose(probability.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("association probabilities must sum to one")
    wrong_cost = _finite_nonnegative(wrong_cost, "wrong_cost")
    unknown_cost = _finite_nonnegative(unknown_cost, "unknown_cost")
    losses = wrong_cost * (1 - probability[1:])
    if losses.size:
        index = int(np.argmin(losses))
        # Decimal thresholds such as 10*(1-0.9)=1 can differ by a few ULPs.
        # Resolve machine-precision ties toward unknown as the contract states.
        tie_tolerance = 8 * np.finfo(np.float64).eps * max(unknown_cost, float(losses[index]))
        if losses[index] < unknown_cost - tie_tolerance:
            return IdentityRisk("accept", index, float(losses[index]), _readonly(losses), unknown_cost)
    return IdentityRisk("unknown", None, unknown_cost, _readonly(losses), unknown_cost)


@dataclass(frozen=True)
class UnknownCostSchedule:
    """Explicit example stage cost, not a calibrated/optimal recovery objective.

    C_U = base + rate * unknown_duration + pressure * scale/(remaining+scale).
    Unknown duration and remaining deadline are causal runtime quantities.
    The bounded deadline term avoids a singularity at zero remaining time.
    A horizon planner still requires elapsed-time and terminal retention losses.
    """

    base: float
    per_unknown_second: float
    deadline_pressure: float
    deadline_time_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("base", "per_unknown_second", "deadline_pressure", "deadline_time_scale"):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        if self.deadline_time_scale <= 0:
            raise ValueError("deadline_time_scale must be positive")

    def __call__(self, unknown_duration: float, remaining_time: float) -> float:
        duration = _finite_nonnegative(unknown_duration, "unknown_duration")
        remaining = _finite_nonnegative(remaining_time, "remaining_time")
        if remaining >= self.deadline_time_scale:
            ratio = self.deadline_time_scale / remaining
            deadline_fraction = ratio / (1 + ratio)
        else:
            deadline_fraction = 1 / (1 + remaining / self.deadline_time_scale)
        cost = self.base + self.per_unknown_second * duration + self.deadline_pressure * deadline_fraction
        if not np.isfinite(cost):
            raise ValueError("unknown cost overflow")
        return cost
