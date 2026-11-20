"""Frozen macro-view predictor; model-free fallback and explicit delivery gate."""
import hashlib,json,math
from pathlib import Path
import numpy as np

class ViewPolicy:
    def __init__(self,root):
        root=Path(root);receipt=json.loads((root/'freeze_receipt.json').read_text())
        assert hashlib.sha256((root/'model.json').read_bytes()).hexdigest()==receipt['model_sha256']
        self.model=json.loads((root/'model.json').read_text());self.scene=self.model['schema']=='river.scene_view_macro.v1'
        self.projection=None
        if self.scene:
            assert hashlib.sha256((root/'projection.npz').read_bytes()).hexdigest()==receipt['projection_sha256']
            with np.load(root/'projection.npz',allow_pickle=False) as z:self.projection=z['projection']

    def select(self,features,*,source_step,capture_ns,delivered_ns,decision_ns,decision_deadline_ns,region_features=None):
        fallback=dict(direction='Left',scores=None,reason='unsupported_or_not_yet_delivered',fallback=True)
        if source_step!=184 or not (0<=capture_ns<=delivered_ns<=decision_ns) or delivered_ns>decision_deadline_ns:return fallback
        try:
            x=np.array([round(float(features[f]),self.model['scalar_round_decimals']) for f in self.model['feature_fields']])
            if self.scene:
                a=np.asarray(region_features,dtype=float)
                if a.shape!=(4,1024) or not np.isfinite(a).all():return fallback
                v=a.reshape(-1)@self.projection;norm=float(np.linalg.norm(v))
                if norm<=1e-12:return fallback
                v=v/norm;x=np.r_[x,v]
            if not np.isfinite(x).all():return fallback
            scores={}
            for action,m in self.model['models'].items():
                scores[action]=float(np.clip(m['intercept']+((x-np.array(m['mean']))/np.array(m['scale']))@np.array(m['weight']),0,1))
            direction=max(self.model['tie_break_order'],key=lambda a:scores[a])
            return dict(direction=direction,scores=scores,reason='frozen_macro_view_regression',fallback=False,
                value_scope='expected factual recovery fraction, not calibrated posterior')
        except (ValueError,TypeError,KeyError):return fallback
