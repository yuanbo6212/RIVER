"""Required controls, not new methods: right, observed coverage, no action prior.

The zero-prior arm runs the same model for matched compute but DISCARDs its
proposal before decoding. Real and decoded chunks are separately archived.
"""
import numpy as np
from calibrated_route_adapter_20261014 import CalibratedRoute, configure
from live_neural_route_20261016 import current_observation
from visibility_search_actions_20261011 import action_residual, right_motion

METHODS=('reference_search_right','reference_candidate_visibility','reference_zero_prior')


class RequiredRoute(CalibratedRoute):
    def request_prior(self,payload):
        response=super().request_prior(payload)
        raw=np.asarray(response['normalized_actions'],float)
        if raw.shape!=(8,5) or not np.isfinite(raw).all():
            raise ValueError('Invalid real prior response')
        self.real_prior_chunk=raw.tolist()
        if self.control_mode=='reference_zero_prior':
            return dict(response,normalized_actions=np.zeros((8,5)).tolist())
        return response

    def residual(self,index,pose,candidates,selected,identity):
        if self.control_mode=='reference_candidate_visibility':
            for c in candidates:
                if set(c)-{'xyxy','confidence','class_id','mask'}:
                    raise ValueError('Non-observation candidate input')
            move,branch=action_residual('causal_visibility',index,candidates,None,{},
                pose,self.origin,self.width,self.height,self.fov)
            branch.update(branch=self.control_mode,neural_pointing_used=False,
                score_is_information_gain=False,identity_score_is_probability=False)
            return move,branch
        move,branch=super().residual(index,pose,candidates,selected,identity)
        if self.control_mode=='reference_search_right':
            vector=right_motion(self.origin)
            move.update(x=float(vector[0]),y=float(vector[1]),z=0.)
            branch.update(translation_candidate_indices=[],translation_bearings_from_origin_deg=[],
                translation_direction=1.,translation_direction_changed=False,
                translation_identity_conditioned=False,fixed_right_translation=True)
        branch.update(branch=self.control_mode,neural_pointing_used=False)
        return move,branch


class ReferenceRequiredControl:
    def __init__(self,url,token,scale,mode,origin,sensor_attributes,infer=None):
        if mode not in METHODS:raise ValueError('Unregistered required control')
        self.impl=RequiredRoute(url,token,scale,'non_goal_away_route',origin,infer)
        configure(self.impl,sensor_attributes)
        self.impl.control_mode=mode
        self.mode,self.next_step=mode,0

    @property
    def path_length(self):return self.impl.path_length

    def observe(self,jpeg,step,pose,initial_box,candidates,selected,identity,neural_reply):
        if step!=self.next_step:raise ValueError('Unordered RGB')
        obs=current_observation(neural_reply,jpeg,step)
        command,trace=self.impl.observe(jpeg,step,pose,initial_box,candidates,selected,identity)
        self.next_step+=1
        if command is not None:
            decoded=trace['pi05_chunk']
            trace=dict(trace,reason=self.mode,pi05_chunk=self.impl.real_prior_chunk,
                action_decode_chunk=decoded,prior_consumed_by_action=self.mode!='reference_zero_prior',
                prior_model_executed_for_matched_compute=True,
                zero_prior_is_action_ablation_not_inference_speedup=self.mode=='reference_zero_prior',
                current_neural_observation=obs,neural_recovery_consumed_by_pointing=False,
                native_neural_pid_executed=False,shared_dual_observer_stack=True,
                action_camera_intrinsics=dict(self.impl.intrinsics),
                calibration_source='actual_sensor_attributes')
        return command,trace
