"""Never-scene control using the same frozen cheap-action risk heads."""
from river_e03_metadecision_v1 import MacroRiskModel as Original
POLICIES=('J3-cheap-risk',)

class MacroRiskModel(Original):
    def select(self,policy,features,**kw):
        if policy not in POLICIES:raise ValueError('cheap control policy required')
        # Same two-head prediction as the independent controller's skip branch.
        kw=dict(kw);kw['remaining_calls']=0
        out=super().select('J3-independent',features,**kw)
        assert not out['compute'] and out['option'] in ('adaptive_left','adaptive_right')
        return dict(out,policy=policy,value_scope='fixed never-scene control, shared cheap-action risk heads')
