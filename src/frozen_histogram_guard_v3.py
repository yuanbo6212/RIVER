"""Preserve raw out-of-domain scores; defer to the existing unsupported reader."""
import json
import time
from pathlib import Path
import numpy as np

EXACT_ERROR = 'score outside frozen histogram range; clipping is forbidden'
ACTIVE_GUARDS = []


class HistogramDomainProxy:
    def __init__(self, model, trace_path=None):
        self.original = model
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.step = None
        self.events = []
        self.predictions = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def predict(self, scores):
        self.predictions += 1
        try:
            return self.original.predict(scores)
        except ValueError as error:
            if str(error) != EXACT_ERROR:
                raise
            values = np.asarray(scores, dtype=float)
            # The original model has already validated shape and finiteness.
            if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
                raise
            event = dict(processed_ordinal=self.step, raw_scores=values.tolist(),
                error=str(error), mono_ns=time.monotonic_ns(), clipping_performed=False,
                candidate_scores_modified=False, output='upstream_null_only',
                downstream='unchanged SharedIdentityReadout validates raw scores independently')
            self.events.append(event)
            if self.trace_path is not None:
                with self.trace_path.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(event, allow_nan=False)+'\n')
            probabilities = np.zeros(len(values)+1)
            probabilities[0] = 1.
            return probabilities


def make_guarded_identity(base, trace_path=None):
    class GuardedCausalRepresentativeIdentity(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = HistogramDomainProxy(self.model, trace_path)
            ACTIVE_GUARDS.append(self.model)

        def update(self, rgb, candidates, step, pose, initial_box=None, fov=90.):
            self.model.step = step
            before = len(self.model.events)
            selected, detail = super().update(rgb, candidates, step, pose, initial_box, fov)
            if len(self.model.events) > before:
                detail = dict(detail, numerical_domain_guard_v3=self.model.events[-1])
            return selected, detail
    return GuardedCausalRepresentativeIdentity


def install(trace_path):
    import river_e02_control
    original = river_e02_control.CausalRepresentativeIdentity
    if getattr(original, '__guard_v3__', False):
        raise RuntimeError('domain guard already installed')
    guarded = make_guarded_identity(original, trace_path)
    guarded.__guard_v3__ = True
    river_e02_control.CausalRepresentativeIdentity = guarded
    return guarded


def summary():
    return dict(instances=len(ACTIVE_GUARDS),
        model_predictions=sum(g.predictions for g in ACTIVE_GUARDS),
        exact_domain_rejections=sum(len(g.events) for g in ACTIVE_GUARDS),
        clipping_performed=False, raw_score_modifications=0,
        shared_readout_model_patched=False)
