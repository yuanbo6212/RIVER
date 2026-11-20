"""Dispatch frozen controls unchanged; common trace fields, no new mechanism."""
import numpy as np
from reference_agreement_control_20261016 import ReferenceAgreementControl
from reference_required_controls_20261016 import ReferenceRequiredControl

METHODS=('reference_non_goal','reference_search_left','reference_search_right',
         'reference_candidate_visibility','reference_zero_prior')


class DevelopmentExtensionControl:
    def __init__(self,url,token,scale,mode,origin,sensor_attributes,infer=None):
        if mode not in METHODS:raise ValueError('Unregistered development control')
        cls=ReferenceAgreementControl if mode in METHODS[:2] else ReferenceRequiredControl
        self.core=cls(url,token,scale,mode,origin,sensor_attributes,infer)
        self.mode=mode

    @property
    def path_length(self):return self.core.path_length

    def observe(self,*args,**kwargs):
        command,trace=self.core.observe(*args,**kwargs)
        if command is not None:
            decoded=trace.get('action_decode_chunk',trace['pi05_chunk'])
            trace=dict(trace,action_decode_chunk=decoded,
                prior_consumed_by_action=self.mode!='reference_zero_prior',
                prior_model_executed_for_matched_compute=True,
                extension_dispatch_only_commands_unchanged=True)
            if self.mode=='reference_zero_prior':
                assert np.array_equal(np.asarray(decoded),np.zeros((8,5)))
        return command,trace
