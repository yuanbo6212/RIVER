"""Three fixed current-candidate readouts, shared frozen causal identity inputs."""
from __future__ import annotations
import json
import numpy as np
from river_e03_runtime_readout import SharedIdentityReadout

ARMS=('full_set','no_empty','simple_margin')


class RejectionReadout(SharedIdentityReadout):
    def __init__(self,bundle_path,*,allow_development_use=False,rejection_arm='full_set'):
        if rejection_arm not in ARMS:raise ValueError('frozen rejection comparison only')
        super().__init__(bundle_path,allow_development_use=allow_development_use)
        self.rejection_arm=rejection_arm

    def predict(self,snapshot,mode,deep_output=None):
        if mode!='cheap' or deep_output is not None:
            raise ValueError('identity rejection comparison uses only current cheap evidence')
        base=super().predict(snapshot,mode)
        if self.rejection_arm=='full_set':return base
        n=len(snapshot.candidates)
        selected=None
        if base['supported'] and n:
            if self.rejection_arm=='no_empty':
                selected=int(np.argmax(np.asarray(base['probabilities'])[1:]))
            else:
                rows={r['index']:r for r in json.loads(snapshot.identity_json)['candidate_scores']}
                selected=min(range(n),key=lambda i:(-rows[i]['positive_similarity'],
                    -snapshot.candidates[i].confidence,i))
                if not rows[selected]['positive_similarity']>rows[selected]['known_negative_similarity']:
                    selected=None
        # Every downstream argmax must read the same declared choice. This is
        # a hard categorical decision encoding; it is not a calibrated score.
        p=np.zeros(n+1);p[0 if selected is None else selected+1]=1.
        out=dict(base,probabilities=p.tolist(),selected_index=selected,
            reason='fixed_rejection_ablation:'+self.rejection_arm+':'+base['reason'],
            runtime_mode='rejection_'+self.rejection_arm,
            probability_scope='onehot_encoding_of_fixed_current_candidate_decision',
            original_frozen_probabilities=base['probabilities'],rejection_arm=self.rejection_arm,
            margin_threshold=0. if self.rejection_arm=='simple_margin' else None)
        if selected is None and n==0:out['reason']+=':no_candidates'
        return out
