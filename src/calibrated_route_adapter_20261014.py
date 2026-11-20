"""Prospective calibration repair, not a new exploration mechanism.

Frozen legacy controllers stay untouched. Only observed-pixel geometry uses
the actual sensor FOV; clocks, priors, readout, thresholds and budgets remain.
"""
import math
import numpy as np
from sensor_camera_contract_20261014 import intrinsic_from_attributes
from task_search_actions_20261014 import LiveTaskSearchControl
from identity_route_actions_20261014 import LiveIdentityRouteControl
from visibility_search_actions_20261011 import coverage_pointing,search_yaw_target,right_motion
from causal_identity_view_v4 import rotation,pixel_ray

METHODS=('fixed_orbit_left','fixed_orbit_right','target_servo_no_negative','all_candidate_away_route','non_goal_away_route')

def configure(policy,attributes):
    K=intrinsic_from_attributes(attributes)
    width,height=int(attributes['image_size_x']),int(attributes['image_size_y'])
    if (width,height)!=(640,360):raise ValueError('This fixed camera contract does not resize observations')
    policy.width,policy.height,policy.fov=width,height,float(attributes['fov'])
    policy.intrinsics=dict(width=width,height=height,fov=policy.fov,fx=float(K[0,0]),fy=float(K[1,1]))

class CalibratedServo(LiveTaskSearchControl):
    def residual(self,index,pose,candidates,selected,identity):
        # Exact target-servo branch; only the physical focal length changes.
        for c in candidates:
            if set(c)-{'xyxy','confidence','class_id','mask'}:raise ValueError('Non-observation candidate input')
        if selected is not None and not 0<=selected<len(candidates):raise ValueError('Identity/candidate mismatch')
        evidence=next((x['positive_minus_negative'] for x in identity['candidate_scores'] if x['index']==selected),None)
        if evidence is not None and not np.isfinite(evidence):raise ValueError('Invalid identity evidence')
        point=[candidates[selected]['xyxy']] if selected is not None else []
        branch=dict(branch=self.mode,candidate_observation_consumed=True,identity_state_consumed=True,
            negative_evidence_consumed=False,mask_readout_changed=False,readout_selected_index=selected,
            appearance_difference=evidence,pointing_candidate_indices=[selected] if point else [],
            score_is_information_gain=False,identity_score_is_probability=False)
        if point:
            yaw,pitch=coverage_pointing(point,pose,self.width,self.height,self.fov)
            branch['phase']='current_goal_candidate_pointing'
        else:
            target=self.origin['yaw']+search_yaw_target(index)
            yaw=float(np.clip((target-pose['yaw']+180.)%360.-180.,-2.,2.));pitch=0.
            branch.update(phase='continue_task_search',desired_yaw=target,visible_candidates_do_not_imply_goal_found=bool(candidates))
        move=-right_motion(self.origin)
        return dict(x=float(move[0]),y=float(move[1]),z=0.,yaw=yaw,pitch=pitch),branch

class CalibratedRoute(LiveIdentityRouteControl):
    def residual(self,index,pose,candidates,selected,identity):
        if not 0<=index<29:raise ValueError('Outside causal action clock')
        for c in candidates:
            if set(c)-{'xyxy','confidence','class_id','mask'}:raise ValueError('Non-observation candidate input')
            b=np.asarray(c['xyxy'],float)
            if b.shape!=(4,) or not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]:raise ValueError('Invalid current observed box')
        if selected is not None and not 0<=selected<len(candidates):raise ValueError('Identity/candidate mismatch')
        score=next((x['positive_minus_negative'] for x in identity['candidate_scores'] if x['index']==selected),None)
        if score is not None and not np.isfinite(score):raise ValueError('Invalid appearance evidence')
        compatible=selected is not None and score is not None and score>0.
        if compatible:
            yaw,pitch=coverage_pointing([candidates[selected]['xyxy']],pose,self.width,self.height,self.fov)
            phase='current_goal_candidate_pointing'
        else:
            target=self.origin['yaw']+search_yaw_target(index)
            yaw=float(np.clip((target-pose['yaw']+180.)%360.-180.,-2.,2.));pitch=0.;phase='continue_task_search'
        used=list(range(len(candidates))) if self.mode=='all_candidate_away_route' else ([selected] if selected is not None and score is not None and score<=0. else [])
        bearings=[]
        for i in used:
            b=candidates[i]['xyxy'];ray=rotation(pose)@pixel_ray((b[0]+b[2])/2.,(b[1]+b[3])/2.,self.width,self.height,self.fov)
            bearings.append((math.degrees(math.atan2(ray[1],ray[0]))-self.origin['yaw']+180.)%360.-180.)
        old=self.direction
        if bearings:
            bearing=float(np.mean(bearings))
            if bearing!=0.:self.direction=-1. if bearing>0. else 1.
        move=right_motion(self.origin)*self.direction
        branch=dict(branch=self.mode,phase=phase,pointing_compatible=compatible,appearance_difference=score,
            readout_selected_index=selected,pointing_candidate_indices=[selected] if compatible else [],translation_candidate_indices=used,
            translation_bearings_from_origin_deg=bearings,translation_direction=self.direction,translation_direction_changed=self.direction!=old,
            translation_identity_conditioned=self.mode=='non_goal_away_route',identity_state_consumed=True,candidate_observation_consumed=True,
            mask_readout_changed=False,hidden_goal_location_inferred=False,score_is_information_gain=False,identity_score_is_probability=False,
            visible_candidates_do_not_imply_goal_found=bool(candidates) and not compatible)
        return dict(x=float(move[0]),y=float(move[1]),z=0.,yaw=yaw,pitch=pitch),branch

class LiveCalibratedRouteControl:
    def __init__(self,url,token,scale,mode,origin,sensor_attributes,infer=None):
        if mode not in METHODS:raise ValueError('Unregistered existing method')
        cls=CalibratedServo if mode=='target_servo_no_negative' else CalibratedRoute
        self.impl=cls(url,token,scale,mode,origin,infer)
        configure(self.impl,sensor_attributes)
    @property
    def path_length(self):return self.impl.path_length
    def observe(self,*args,**kwargs):
        command,trace=self.impl.observe(*args,**kwargs)
        if command is not None:trace=dict(trace,action_camera_intrinsics=dict(self.impl.intrinsics),calibration_source='actual_sensor_attributes',new_method_mechanism=False)
        return command,trace
