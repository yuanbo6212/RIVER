"""Appearance-only development calibration infrastructure (NumPy + stdlib).

No dataset is fitted on import. Score density ratios are not spatial likelihoods,
class posteriors, detection probabilities, or mutually exclusive associations.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np


def _array(value: Any, dtype=float) -> np.ndarray:
    data = np.asarray(value, dtype=dtype)
    # Immutable bytes backing also prevents setflags(write=True).
    return np.frombuffer(data.tobytes(), dtype=data.dtype).reshape(data.shape)


def _ids(values: Sequence[str], name: str, unique: bool = False) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of IDs, not one string")
    result = tuple(values)
    if not result or any(not isinstance(x, str) or not x.strip() for x in result):
        raise ValueError(f"{name} must contain nonempty string IDs")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ReservedGroups:
    group_ids: Tuple[str, ...]
    source_sha256: str
    origin: str

    def __post_init__(self):
        object.__setattr__(self, "group_ids", tuple(sorted(_ids(self.group_ids, "reserved groups", True))))
        if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256):
            raise ValueError("reservation source must have a lowercase SHA-256 digest")
        if self.origin not in {"historical_test_csv", "synthetic"}:
            raise ValueError("unsupported reservation origin")


def load_historical_test_groups(path: Any) -> ReservedGroups:
    """Read group keys only; bind exclusion to the actual 23-group source hash."""
    source = Path(path)
    raw = source.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    if not rows or "group" not in rows[0]:
        raise ValueError("historical test source must contain a group column")
    groups = tuple(sorted({row["group"] for row in rows}))
    if len(groups) != 23:
        raise ValueError("historical source must identify exactly 23 unique test groups")
    return ReservedGroups(groups, hashlib.sha256(raw).hexdigest(), "historical_test_csv")


@dataclass(frozen=True)
class GroupProtocol:
    fit_groups: Tuple[str, ...]
    validation_groups: Tuple[str, ...]
    reserved: ReservedGroups
    data_kind: str = "synthetic"
    additional_reserved_groups: Tuple[str, ...] = ()

    def __post_init__(self):
        fit = _ids(self.fit_groups, "fit groups", True)
        validation = _ids(self.validation_groups, "validation groups", True)
        extra = tuple(self.additional_reserved_groups)
        if extra:
            extra = _ids(extra, "additional reserved groups", True)
        if self.data_kind not in {"synthetic", "development"}:
            raise ValueError("only synthetic or authorized development fitting is supported")
        if self.data_kind == "development" and self.reserved.origin != "historical_test_csv":
            raise ValueError("development fitting requires the actual historical-test CSV reservation")
        reserved = set(self.reserved.group_ids) | set(extra)
        if set(fit) & set(validation) or set(fit) & reserved or set(validation) & reserved:
            raise ValueError("fit, validation, and reserved test groups must be disjoint")
        object.__setattr__(self, "fit_groups", tuple(sorted(fit)))
        object.__setattr__(self, "validation_groups", tuple(sorted(validation)))
        object.__setattr__(self, "additional_reserved_groups", tuple(sorted(extra)))

    def check(self, groups: Sequence[str], role: str) -> Tuple[str, ...]:
        actual = _ids(groups, "sample groups")
        if role not in {"fit", "validation"}:
            raise ValueError("role must be fit or validation; test fitting/evaluation is not enabled here")
        allowed = set(self.fit_groups if role == "fit" else self.validation_groups)
        if not set(actual) <= allowed:
            raise ValueError(f"sample groups are outside the declared {role} partition")
        return actual


@dataclass(frozen=True)
class HistogramSpec:
    feature_names: Tuple[str, ...]
    bin_edges: Tuple[np.ndarray, ...]
    pseudocount: float
    observation_definition: str

    def __post_init__(self):
        names = _ids(self.feature_names, "feature names", True)
        edges = tuple(_array(x) for x in self.bin_edges)
        if len(edges) != len(names):
            raise ValueError("one fixed bin-edge array is required per feature")
        for edge in edges:
            with np.errstate(over="ignore", invalid="ignore"):
                widths = np.diff(edge) if edge.ndim == 1 else np.array([])
            if edge.ndim != 1 or edge.size < 2 or not np.all(np.isfinite(edge)) or not np.all(np.isfinite(widths)) or np.any(widths <= 0):
                raise ValueError("bin edges must be finite and strictly increasing")
        alpha = float(self.pseudocount)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("a finite positive pseudocount per joint bin is required")
        if not isinstance(self.observation_definition, str) or not self.observation_definition.strip():
            raise ValueError("the crop/score/reference observation definition is required")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(self, "pseudocount", alpha)


def _scores(scores: Any, edges: Sequence[np.ndarray]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(edges):
        raise ValueError("scores must be nonempty N x number-of-selected-features")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    for axis, edge in enumerate(edges):
        if np.any(values[:, axis] < edge[0]) or np.any(values[:, axis] > edge[-1]):
            raise ValueError("scores outside the declared range are rejected, never silently clipped")
    return values


@dataclass(frozen=True)
class LabeledAppearanceBatch:
    scores: np.ndarray
    is_target: np.ndarray
    group_ids: Tuple[str, ...]
    frame_ids: Tuple[str, ...]
    sample_ids: Tuple[str, ...]

    def __post_init__(self):
        scores = _array(self.scores)
        labels = np.asarray(self.is_target)
        if scores.ndim != 2 or scores.shape[0] == 0:
            raise ValueError("scores must be a nonempty N x D array")
        if labels.shape != (scores.shape[0],) or labels.dtype.kind != "b":
            raise ValueError("is_target must be a boolean vector, not soft/classifier scores")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "is_target", _array(labels, bool))
        for name in ("group_ids", "frame_ids", "sample_ids"):
            ids = _ids(getattr(self, name), name, unique=name == "sample_ids")
            if len(ids) != scores.shape[0]:
                raise ValueError(f"{name} must identify every candidate row")
            object.__setattr__(self, name, ids)


@dataclass(frozen=True)
class ClutterFrameExposure:
    """Complete fit-frame ledger, including zero-candidate/zero-clutter frames."""
    frame_ids: Tuple[str, ...]
    group_ids: Tuple[str, ...]
    non_target_counts: Tuple[int, ...]

    def __post_init__(self):
        frames = _ids(self.frame_ids, "exposure frames", True)
        groups = _ids(self.group_ids, "exposure groups")
        counts = tuple(self.non_target_counts)
        if len(groups) != len(frames) or len(counts) != len(frames):
            raise ValueError("exposure ledger columns must have equal lengths")
        if any(isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer)) or c < 0 for c in counts):
            raise ValueError("clutter counts must be nonnegative integers")
        object.__setattr__(self, "frame_ids", frames)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "non_target_counts", tuple(int(c) for c in counts))


@dataclass(frozen=True)
class FitMetadata:
    schema: str
    data_id: str
    fit_data_sha256: str
    model_sha256: str
    fit_groups: Tuple[str, ...]
    validation_groups: Tuple[str, ...]
    reserved_groups: Tuple[str, ...]
    reserved_source_sha256: str
    data_kind: str
    target_rows: int
    clutter_rows: int
    exposure_frames: int


@dataclass(frozen=True)
class AppearanceRatio:
    """Appearance measure only. No implicit conversion to belief evidence exists."""
    target_score_density: np.ndarray
    clutter_score_density: np.ndarray
    clutter_score_intensity: np.ndarray
    log_conditional_density_ratio: np.ndarray
    log_appearance_to_clutter_intensity_ratio: np.ndarray
    feature_names: Tuple[str, ...]
    model_sha256: str


@dataclass(frozen=True)
class HistogramAppearanceModel:
    spec: HistogramSpec
    target_bin_mass: np.ndarray
    clutter_bin_mass: np.ndarray
    expected_clutter_count: float
    metadata: FitMetadata

    def __post_init__(self):
        shape = tuple(len(edge) - 1 for edge in self.spec.bin_edges)
        for name in ("target_bin_mass", "clutter_bin_mass"):
            mass = _array(getattr(self, name))
            if mass.shape != shape or not np.all(np.isfinite(mass)) or np.any(mass <= 0) or not np.isclose(mass.sum(), 1):
                raise ValueError("class-conditional bin masses must be strictly positive and normalized")
            object.__setattr__(self, name, mass)
        rate = float(self.expected_clutter_count)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("expected clutter count must be finite and positive")
        object.__setattr__(self, "expected_clutter_count", rate)

    def evaluate(self, scores: Any, feature_names: Optional[Sequence[str]] = None) -> AppearanceRatio:
        selected = self.spec.feature_names if feature_names is None else _ids(feature_names, "selected features", True)
        if not set(selected) <= set(self.spec.feature_names):
            raise ValueError("unknown feature in likelihood query")
        axes = tuple(self.spec.feature_names.index(name) for name in selected)
        remaining = tuple(i for i in range(len(self.spec.feature_names)) if i not in axes)
        edges = tuple(self.spec.bin_edges[i] for i in axes)
        values = _scores(scores, edges)
        indices = tuple(np.minimum(np.searchsorted(edge, values[:, i], side="right") - 1, len(edge) - 2) for i, edge in enumerate(edges))
        def density(mass):
            ordered = np.transpose(mass, axes + remaining)
            if remaining:
                ordered = ordered.sum(axis=tuple(range(len(axes), ordered.ndim)))
            log_volume = sum(np.log(np.diff(edge))[index] for edge, index in zip(edges, indices))
            return np.log(ordered[indices]) - log_volume
        log_target = density(self.target_bin_mass)
        log_clutter = density(self.clutter_bin_mass)
        log_intensity = np.log(self.expected_clutter_count) + log_clutter
        with np.errstate(over="ignore", under="ignore"):
            target, clutter, intensity = np.exp(log_target), np.exp(log_clutter), np.exp(log_intensity)
        if any(not np.all(np.isfinite(a)) or np.any(a <= 0) for a in (target, clutter, intensity)):
            raise ValueError("declared ranges/density values exceed representable positive float64 values")
        return AppearanceRatio(*(_array(a) for a in (target, clutter, intensity, log_target - log_clutter, log_target - log_intensity)), selected, self.metadata.model_sha256)

    def conditional_log_ratio_increment(self, full_scores: Any, known_features: Sequence[str]) -> np.ndarray:
        """Extra same-crop feature evidence conditional on known features.

        Derives the known marginal from THIS joint histogram. Do not combine
        separately fitted marginal histograms as if same-crop features independent.
        Lambda is charged in the base ratio only, not in this increment.
        """
        values = _scores(full_scores, self.spec.bin_edges)
        known = _ids(known_features, "known features", True)
        if not set(known) <= set(self.spec.feature_names):
            raise ValueError("unknown conditioning feature")
        columns = [self.spec.feature_names.index(name) for name in known]
        joint = self.evaluate(values).log_conditional_density_ratio
        base = self.evaluate(values[:, columns], known).log_conditional_density_ratio
        return _array(joint - base)


def fit_appearance_histogram(
    batch: LabeledAppearanceBatch, exposure: ClutterFrameExposure, *,
    spec: HistogramSpec, protocol: GroupProtocol, data_id: str,
) -> HistogramAppearanceModel:
    """Fit only the declared fit groups, using fixed bins and positive smoothing."""
    if not isinstance(data_id, str) or not data_id.strip():
        raise ValueError("an auditable data version/ID is required")
    values = _scores(batch.scores, spec.bin_edges)
    protocol.check(batch.group_ids, "fit")
    protocol.check(exposure.group_ids, "fit")
    if not np.any(batch.is_target) or np.all(batch.is_target):
        raise ValueError("both target and non-target score samples are required")
    ledger = dict(zip(exposure.frame_ids, zip(exposure.group_ids, exposure.non_target_counts)))
    observed_clutter = {frame: 0 for frame in ledger}
    target_count = {frame: 0 for frame in ledger}
    for frame, group, target in zip(batch.frame_ids, batch.group_ids, batch.is_target):
        if frame not in ledger or ledger[frame][0] != group:
            raise ValueError("candidate frame/group is inconsistent with exposure ledger")
        target_count[frame] += int(target)
        observed_clutter[frame] += int(not target)
    if any(target_count[f] > 1 for f in ledger):
        raise ValueError("at-most-one retained target candidate per frame is required")
    if any(observed_clutter[f] != ledger[f][1] for f in ledger):
        raise ValueError("clutter score rows must match every exposure-frame count, including zeros")
    rate = sum(exposure.non_target_counts) / len(exposure.frame_ids)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("positive finite observed clutter exposure rate is required")
    counts = [np.histogramdd(values[mask], bins=spec.bin_edges)[0] for mask in (batch.is_target, ~batch.is_target)]
    masses = []
    for count in counts:
        with np.errstate(over="ignore", invalid="ignore"):
            smoothed = count + spec.pseudocount
            mass = smoothed / smoothed.sum()
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
            raise ValueError("smoothing configuration does not yield finite positive bin masses")
        masses.append(mass)
    data_hash = _digest(dict(scores=values.tolist(), target=batch.is_target.tolist(), groups=batch.group_ids, frames=batch.frame_ids, samples=batch.sample_ids, exposure=list(zip(exposure.frame_ids, exposure.group_ids, exposure.non_target_counts))))
    reserved = tuple(sorted(set(protocol.reserved.group_ids) | set(protocol.additional_reserved_groups)))
    fingerprint = dict(schema="river-appearance-histogram-v1", data_id=data_id, data_sha256=data_hash, feature_names=spec.feature_names, edges=[edge.tolist() for edge in spec.bin_edges], pseudocount=spec.pseudocount, observation_definition=spec.observation_definition, target_mass=masses[0].tolist(), clutter_mass=masses[1].tolist(), expected_clutter_count=rate, fit_groups=protocol.fit_groups, validation_groups=protocol.validation_groups, reserved_groups=reserved, reserved_source_sha256=protocol.reserved.source_sha256, data_kind=protocol.data_kind)
    metadata = FitMetadata(fingerprint["schema"], data_id, data_hash, _digest(fingerprint), protocol.fit_groups, protocol.validation_groups, reserved, protocol.reserved.source_sha256, protocol.data_kind, int(batch.is_target.sum()), int((~batch.is_target).sum()), len(exposure.frame_ids))
    return HistogramAppearanceModel(spec, masses[0], masses[1], rate, metadata)


def class_posterior_from_log_density_ratio(log_ratio: Any, target_class_prior: float) -> np.ndarray:
    """Candidate classification only: prior odds * f_target/f_clutter.

    The class prior refers to the candidate sampling population, not pD, null
    association, or an arbitrary balance of training examples.
    """
    ratio = np.asarray(log_ratio, dtype=float)
    prior = float(target_class_prior)
    if not np.all(np.isfinite(ratio)) or not np.isfinite(prior) or not 0 <= prior <= 1:
        raise ValueError("finite log ratios and a class prior in [0,1] are required")
    if prior in (0, 1):
        return _array(np.full_like(ratio, prior))
    log_odds = ratio + np.log(prior) - np.log1p(-prior)
    result = np.empty_like(log_odds)
    positive = log_odds >= 0
    result[positive] = 1 / (1 + np.exp(-log_odds[positive]))
    exp_negative = np.exp(log_odds[~positive])
    result[~positive] = exp_negative / (1 + exp_negative)
    return _array(result)


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_probability: Optional[float]
    observed_frequency: Optional[float]


@dataclass(frozen=True)
class ProbabilityMetrics:
    sample_count: int
    group_ids: Tuple[str, ...]
    brier: float
    nll: float
    ece: float
    accuracy: float
    reliability_bins: Tuple[ReliabilityBin, ...]
    definition: str


def _reliability(confidence: np.ndarray, outcome: np.ndarray, n_bins: int):
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer fixed before evaluation")
    edges = np.linspace(0, 1, n_bins + 1)
    assignments = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, n_bins - 1)
    bins, ece = [], 0.0
    for i in range(n_bins):
        selected = assignments == i
        count = int(selected.sum())
        probability = float(confidence[selected].mean()) if count else None
        frequency = float(outcome[selected].mean()) if count else None
        if count:
            ece += count / confidence.size * abs(probability - frequency)
        bins.append(ReliabilityBin(float(edges[i]), float(edges[i + 1]), count, probability, frequency))
    return ece, tuple(bins)


def evaluate_association_probabilities(
    probabilities: Sequence[Any], labels: Sequence[int], group_ids: Sequence[str], *,
    protocol: GroupProtocol, n_bins: int = 10,
) -> ProbabilityMetrics:
    """Future integration evaluator: each variable-length row is [null,candidates].

    labels use 0 for null and i+1 for target candidate i. Brier is the full
    multiclass sum, NLL is unclipped (zero true mass -> infinity), ECE is top-label.
    """
    groups = protocol.check(group_ids, "validation")
    if len(probabilities) != len(labels) or len(labels) != len(groups):
        raise ValueError("probability rows, association labels, and group IDs must align")
    brier, losses, confidence, correct = [], [], [], []
    for row, label in zip(probabilities, labels):
        p = np.asarray(row, dtype=float)
        if p.ndim != 1 or p.size < 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10):
            raise ValueError("association rows must be normalized finite categorical probabilities including null")
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)) or not 0 <= label < p.size:
            raise ValueError("association label is outside the null/candidate row")
        truth = np.zeros_like(p)
        truth[label] = 1
        brier.append(float(np.sum((p - truth)**2)))
        losses.append(float(-np.log(p[label])) if p[label] > 0 else float("inf"))
        confidence.append(float(p.max()))
        correct.append(int(np.argmax(p) == label))
    confidence, correct = np.asarray(confidence), np.asarray(correct)
    ece, bins = _reliability(confidence, correct, n_bins)
    return ProbabilityMetrics(len(groups), tuple(sorted(set(groups))), float(np.mean(brier)), float(np.mean(losses)), ece, float(correct.mean()), bins, "categorical-null-inclusive; Brier=sum; ECE=top-label")


def evaluate_binary_probabilities(
    probabilities: Any, is_target: Any, group_ids: Sequence[str], *,
    protocol: GroupProtocol, n_bins: int = 10,
) -> ProbabilityMetrics:
    """Evaluate real supplied target-class probabilities, never raw HSV scores."""
    groups = protocol.check(group_ids, "validation")
    p, y = np.asarray(probabilities, dtype=float), np.asarray(is_target)
    if p.shape != (len(groups),) or y.shape != p.shape or y.dtype.kind != "b":
        raise ValueError("binary probabilities, boolean labels, and group IDs must align")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("binary probabilities must be finite and in [0,1]")
    true_mass = np.where(y, p, 1 - p)
    with np.errstate(divide="ignore"):
        nll = float(np.mean(-np.log(true_mass)))
    ece, bins = _reliability(p, y, n_bins)
    return ProbabilityMetrics(len(groups), tuple(sorted(set(groups))), float(np.mean((p - y)**2)), nll, ece, float(np.mean((p >= 0.5) == y)), bins, "candidate-binary; Brier=positive-class; ECE=target-reliability")
