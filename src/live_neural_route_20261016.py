"""Prospective online adapter, not a claimed novel module or tested performance.

Preserve calibrated translation and pi0.5 budgeting. Both adaptive arms point
using the SAME current accepted neural recovery box, or the SAME search clock.
The original reference-appearance identity still supplies negative bearings.
This is explicitly a shared dual-observer stack, not replacement of all HSV
identity by OA-VAT and not execution of OA-VAT's native PID.
"""
import hashlib
import math
import numpy as np
from calibrated_route_adapter_20261014 import CalibratedRoute, configure
from visibility_search_actions_20261011 import coverage_pointing, search_yaw_target

MODES=('fixed_orbit_left','all_candidate_away_route','non_goal_away_route')
SOURCES=('initial_task_box','current_image_reacquisition','current_confident_tracker_adopted_box','no_current_accepted_observation')

def current_observation(reply,jpeg,step):
    allowed={'status','step','jpeg_sha256','observation','delivered_prefix','native_commands','image_sha256','commands_executed'}
    if set(reply)-allowed or reply.get('status')!='frame_complete':
        raise ValueError('Unexpected recovery response, no privileged additions allowed')
    if reply['step']!=step or reply['delivered_prefix']!=step+1 or reply['jpeg_sha256']!=hashlib.sha256(jpeg).hexdigest():
        raise ValueError('Stale, future or different-image recovery response')
    if reply.get('commands_executed',False):
        raise ValueError('This adapter never executes the neural core PID')
    obs=reply['observation']
    if set(obs)-{'step','box_xywh','source','confidence'} or obs['step']!=step or obs['source'] not in SOURCES:
        raise ValueError('Invalid current observation schema')
    box=obs['box_xywh']
    if (box is None)!=(obs['source']=='no_current_accepted_observation'):
        raise ValueError('Missing box and accepted-source mismatch')
    if obs['source']=='initial_task_box' and step!=0:
        raise ValueError('Later task annotation forbidden')
    if obs['source']=='current_image_reacquisition' and step==0:
        raise ValueError('First frame must be the legal initial task')
    if obs['source']=='current_confident_tracker_adopted_box':
        if not math.isfinite(obs.get('confidence',float('nan'))) or obs['confidence']<.5:
            raise ValueError('Not the frozen accepted neural readout')
    if box is not None:
        a=np.asarray(box,float)
        if a.shape!=(4,) or not np.isfinite(a).all() or a[2]<=0 or a[3]<=0:
            raise ValueError('Invalid accepted rectangle')
    return dict(obs)

class NeuralPointingRoute(CalibratedRoute):
    def residual(self,index,pose,candidates,selected,identity):
        move,branch=super().residual(index,pose,candidates,selected,identity)
        obs=self.neural_observation
        if obs['step']!=185+index:
            raise ValueError('Action may not consume a cached observation')
        box=obs['box_xywh']
        branch['legacy_goal_candidate_compatible']=branch['pointing_compatible']
        if box is not None:
            x,y,w,h=box
            yaw,pitch=coverage_pointing([[x,y,x+w,y+h]],pose,self.width,self.height,self.fov)
            phase='current_neural_goal_pointing'
        else:
            desired=self.origin['yaw']+search_yaw_target(index)
            yaw=float(np.clip((desired-pose['yaw']+180.)%360.-180.,-2.,2.));pitch=0.
            phase='continue_neural_unconfirmed_search'
        move['yaw'],move['pitch']=yaw,pitch
        branch.update(phase=phase,pointing_compatible=box is not None,pointing_candidate_indices=[],
                      pointing_source='shared_current_neural_readout',neural_evidence_step=obs['step'],
                      neural_box_xywh=box,neural_source=obs['source'],
                      visible_candidates_do_not_imply_goal_found=bool(candidates) and box is None,
                      negative_translation_identity_still_reference_appearance=True)
        return move,branch

class LiveNeuralRouteControl:
    def __init__(self,url,token,scale,mode,origin,sensor_attributes,infer=None):
        if mode not in MODES:raise ValueError('Unregistered prospective online arm')
        self.impl=NeuralPointingRoute(url,token,scale,mode,origin,infer)
        configure(self.impl,sensor_attributes);self.next_step=0;self.mode=mode
    @property
    def path_length(self):return self.impl.path_length
    def observe(self,jpeg,step,pose,initial_box,candidates,selected,identity,neural_reply):
        if step!=self.next_step:raise ValueError('Unordered observation')
        obs=current_observation(neural_reply,jpeg,step)
        self.impl.neural_observation=obs
        command,trace=self.impl.observe(jpeg,step,pose,initial_box,candidates,selected,identity)
        self.next_step+=1
        if command is not None:
            trace=dict(trace,current_neural_observation=obs,
                       neural_recovery_consumed_by_pointing=self.mode!='fixed_orbit_left',
                       native_neural_pid_executed=False,shared_dual_observer_stack=True,
                       action_camera_intrinsics=dict(self.impl.intrinsics),
                       calibration_source='actual_sensor_attributes',
                       prospective_online_adapter_not_validated_performance=True)
        return command,trace
