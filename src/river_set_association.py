"""Offline appearance-only auxiliary-representative association screen.

One representative is drawn uniformly from all acceptable target proposals in a
positive frame. Other proposals are *nuisance*, including duplicate target
members. This is an auxiliary categorical task, not physical identity membership
or a geometric/state posterior. Nothing here is connected to a policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from river_calibration import (
    ClutterFrameExposure, GroupProtocol, HistogramSpec, LabeledAppearanceBatch,
)


MODEL_VERSION = "river-appearance-uniform-representative-v1"


def _immutable(values: Any) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _scores(scores: Any, spec: HistogramSpec, allow_empty: bool = False) -> np.ndarray:
    a = np.asarray(scores, dtype=float)
    if a.ndim != 2 or a.shape[1] != len(spec.feature_names) or (not allow_empty and not len(a)):
        raise ValueError("scores must have one column per fixed feature")
    if not np.isfinite(a).all():
        raise ValueError("scores must be finite")
    for j, edge in enumerate(spec.bin_edges):
        if np.any(a[:, j] < edge[0]) or np.any(a[:, j] > edge[-1]):
            raise ValueError("score outside frozen histogram range; clipping is forbidden")
    return a


@dataclass(frozen=True)
class RepresentativeAssociationModel:
    spec: HistogramSpec
    target_bin_mass: np.ndarray
    nuisance_bin_mass: np.ndarray
    candidate_present_prior_q: float
    expected_nuisance_count: float
    metadata: dict

    def __post_init__(self):
        shape = tuple(len(e) - 1 for e in self.spec.bin_edges)
        for name in ("target_bin_mass", "nuisance_bin_mass"):
            a = _immutable(getattr(self, name))
            if a.shape != shape or not np.isfinite(a).all() or np.any(a <= 0) or not np.isclose(a.sum(), 1):
                raise ValueError("histogram masses must be positive and normalized")
            object.__setattr__(self, name, a)
        if not np.isfinite(self.candidate_present_prior_q) or not 0 < self.candidate_present_prior_q < 1:
            raise ValueError("q must lie strictly between zero and one")
        if not np.isfinite(self.expected_nuisance_count) or self.expected_nuisance_count <= 0:
            raise ValueError("positive nuisance count is required")
        # Prevent later caller mutations from changing the fitted provenance.
        object.__setattr__(self, "metadata", json.loads(json.dumps(self.metadata, allow_nan=False)))

    def predict(self, scores: Any) -> np.ndarray:
        """Return [null, candidate_0, ...] using scores only, no labels/geometry.

        Conditional on the model, w0=1-q and wj=q*fT(sj)/(lambda*fN(sj)).
        This is a normalized auxiliary-representative probability, not a claim
        that two target-matching proposals cannot both have target identity.
        """
        a = _scores(scores, self.spec, allow_empty=True)
        if not len(a):
            return _immutable([1.0])
        ix = tuple(np.minimum(np.searchsorted(e, a[:, j], side="right") - 1, len(e) - 2)
                   for j, e in enumerate(self.spec.bin_edges))
        # Same-cell volumes cancel in fT/fN. The count intensity appears once.
        log_terms = np.concatenate((
            [np.log1p(-self.candidate_present_prior_q)],
            np.log(self.candidate_present_prior_q) - np.log(self.expected_nuisance_count)
            + np.log(self.target_bin_mass[ix]) - np.log(self.nuisance_bin_mass[ix]),
        ))
        weights = np.exp(log_terms - log_terms.max())
        return _immutable(weights / weights.sum())

    def artifact(self) -> dict:
        return dict(
            model_version=MODEL_VERSION,
            feature_names=list(self.spec.feature_names),
            bin_edges=[e.tolist() for e in self.spec.bin_edges],
            histogram_pseudocount=self.spec.pseudocount,
            observation_definition=self.spec.observation_definition,
            target_bin_mass=self.target_bin_mass.tolist(),
            nuisance_bin_mass=self.nuisance_bin_mass.tolist(),
            candidate_present_prior_q=self.candidate_present_prior_q,
            expected_nuisance_count=self.expected_nuisance_count,
            metadata=self.metadata,
        )


def fit_representative_association(
    batch: LabeledAppearanceBatch, exposure: ClutterFrameExposure, *,
    spec: HistogramSpec, protocol: GroupProtocol, data_id: str,
    q_beta_pseudocount: float = 0.5,
) -> RepresentativeAssociationModel:
    """Fit all proposals/frames using expected Uniform(A) sufficient statistics.

    For k>0 acceptable proposals each gets representative weight 1/k and
    nuisance weight 1-1/k. Every non-target proposal gets nuisance weight 1.
    No candidate is chosen using its IoU and no frame/candidate is dropped.
    q is marginal proposal coverage per archived exposure, NOT pD|visible.
    """
    if not isinstance(data_id, str) or not data_id.strip():
        raise ValueError("data version is required")
    beta = float(q_beta_pseudocount)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("positive fixed beta pseudocount is required")
    values = _scores(batch.scores, spec)
    protocol.check(batch.group_ids, "fit")
    protocol.check(exposure.group_ids, "fit")
    ledger = dict(zip(exposure.frame_ids, zip(exposure.group_ids, exposure.non_target_counts)))
    target_counts = dict.fromkeys(ledger, 0)
    non_target_counts = dict.fromkeys(ledger, 0)
    for frame, group, target in zip(batch.frame_ids, batch.group_ids, batch.is_target):
        if frame not in ledger or group != ledger[frame][0]:
            raise ValueError("candidate frame/group disagrees with exposure")
        target_counts[frame] += int(target)
        non_target_counts[frame] += int(not target)
    if any(non_target_counts[f] != ledger[f][1] for f in ledger):
        raise ValueError("all non-target score rows must match the complete exposure ledger")
    present = sum(k > 0 for k in target_counts.values())
    if not present:
        raise ValueError("at least one target-match frame is required")
    target_weight = np.array([1.0 / target_counts[f] if y else 0.0
                              for f, y in zip(batch.frame_ids, batch.is_target)])
    nuisance_weight = 1.0 - target_weight
    if nuisance_weight.sum() <= 0:
        raise ValueError("observed nuisance exposure is required")
    target_counts_hist = np.histogramdd(values, bins=spec.bin_edges, weights=target_weight)[0]
    nuisance_counts_hist = np.histogramdd(values, bins=spec.bin_edges, weights=nuisance_weight)[0]
    target_mass = (target_counts_hist + spec.pseudocount)
    target_mass /= target_mass.sum()
    nuisance_mass = (nuisance_counts_hist + spec.pseudocount)
    nuisance_mass /= nuisance_mass.sum()
    n = len(exposure.frame_ids)
    q = (present + beta) / (n + 2 * beta)
    rate = float(nuisance_weight.sum() / n)
    input_content = dict(
        scores=values.tolist(), target_membership=batch.is_target.tolist(),
        groups=batch.group_ids, frames=batch.frame_ids, samples=batch.sample_ids,
        exposure=list(zip(exposure.frame_ids, exposure.group_ids, exposure.non_target_counts)),
    )
    metadata = dict(
        version=MODEL_VERSION, data_id=data_id, data_kind=protocol.data_kind,
        fit_data_sha256=_hash(input_content), fit_groups=list(protocol.fit_groups),
        validation_groups=list(protocol.validation_groups),
        reserved_groups=sorted(set(protocol.reserved.group_ids) | set(protocol.additional_reserved_groups)),
        reserved_source_sha256=protocol.reserved.source_sha256,
        fit_frames=n, fit_candidates=len(values), fit_target_match_frames=present,
        fit_duplicate_target_match_frames=sum(k > 1 for k in target_counts.values()),
        fit_physical_target_member_rows=int(batch.is_target.sum()),
        fit_physical_non_target_rows=int((~batch.is_target).sum()),
        fit_physical_non_target_rate=float(sum(non_target_counts.values()) / n),
        fit_representative_effective_count=float(target_weight.sum()),
        fit_nuisance_effective_count=float(nuisance_weight.sum()),
        q_beta_pseudocount=beta,
        q_definition="P(at least one acceptable target proposal in an archived frame); not pD conditioned on visibility",
        target_weight="1/k for each of k acceptable proposals; zero for non-targets",
        nuisance_weight="1 minus representative weight; includes other target members, not physical clutter",
        assumptions=["Bernoulli auxiliary representative", "independent Poisson nuisance count and iid nuisance scores", "exchangeable score-only observations"],
        limitations=["detector duplicates and distractors can violate Poisson/iid assumptions", "marginal archived exposure does not identify visibility-conditioned detection", "no geometry, state, temporal dependence, causal action prediction or policy use"],
        policy_release=False,
    )
    fingerprint = dict(
        metadata=metadata, features=spec.feature_names,
        edges=[e.tolist() for e in spec.bin_edges], pseudocount=spec.pseudocount,
        observation_definition=spec.observation_definition,
        target_mass=target_mass.tolist(), nuisance_mass=nuisance_mass.tolist(), q=q, rate=rate,
    )
    metadata["model_sha256"] = _hash(fingerprint)
    return RepresentativeAssociationModel(spec, target_mass, nuisance_mass, q, rate, metadata)


def simple_positive_selector(scores: Any, positive_column: int = 0) -> np.ndarray:
    """Deterministic baseline: maximum positive-reference similarity, null if empty.

    The returned one-hot vector represents a deterministic decision only. Raw
    similarities have not been interpreted as calibrated probabilities.
    """
    a = np.asarray(scores, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all() or not 0 <= positive_column < a.shape[1]:
        raise ValueError("finite scores and a valid positive column are required")
    p = np.zeros(len(a) + 1)
    p[1 + int(np.argmax(a[:, positive_column])) if len(a) else 0] = 1.0
    return _immutable(p)


def score_acceptable_set(probabilities: Any, acceptable_labels: Sequence[int]) -> dict:
    """Score every acceptable label; never choose one GT-best target proposal.

    Set-mass NLL is a union-event diagnostic, NOT generally a proper categorical
    score for arbitrary overlapping weak-label sets. Uniform-representative NLL
    and expected one-hot Brier are proper for the explicitly declared auxiliary
    categorical target distribution Uniform(A), not physical unique association.
    Null-vs-any-candidate scores use the observed event len(target set)>0.
    """
    p = np.asarray(probabilities, dtype=float)
    labels = tuple(acceptable_labels)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("normalized categorical probabilities including null required")
    if not labels or len(set(labels)) != len(labels) or any(isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)) or not 0 <= j < len(p) for j in labels):
        raise ValueError("acceptable labels must be unique valid indices")
    if 0 in labels and len(labels) != 1:
        raise ValueError("null is acceptable exactly when no target candidate is acceptable")
    a = np.array(labels, dtype=int)
    true_mass = float(p[a].sum())
    with np.errstate(divide="ignore"):
        set_nll = float(-np.log(true_mass))
        uniform_nll = float(-np.log(p[a]).mean())
    truth = np.zeros_like(p)
    truth[a] = 1 / len(a)
    decision = int(np.argmax(p))
    target_present = labels != (0,)
    coverage_probability = float(p[1:].sum())
    coverage_mass = coverage_probability if target_present else float(p[0])
    with np.errstate(divide="ignore"):
        coverage_nll = float(-np.log(coverage_mass))
    wrong_kind = "correct"
    if decision not in labels:
        wrong_kind = "null_on_target_match" if decision == 0 else ("candidate_on_null" if not target_present else "wrong_candidate_on_target_match")
    return dict(
        acceptable_mass=true_mass, acceptable_set_nll=set_nll,
        uniform_representative_nll=uniform_nll,
        uniform_target_brier=float(np.sum((p - truth) ** 2)),
        expected_representative_onehot_brier=float(np.sum(p*p) - 2 * p[a].mean() + 1),
        coverage_probability=coverage_probability,
        candidate_present_brier=float((coverage_probability - int(target_present)) ** 2),
        candidate_present_nll=coverage_nll,
        correct=int(decision in labels), decision=decision, error_kind=wrong_kind,
        top_probability=float(p.max()), target_match_count=len(a) if target_present else 0,
        conditional_acceptable_entropy=(float(-np.sum((p[a] / true_mass) * np.log(p[a] / true_mass))) if true_mass > 0 and np.all(p[a] > 0) else None),
        duplicate_uniformity_excess=(uniform_nll - set_nll - float(np.log(len(a)))) if np.isfinite(uniform_nll) and np.isfinite(set_nll) else None,
    )
