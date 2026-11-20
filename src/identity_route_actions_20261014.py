"""Causal lateral exploration using fixed task-disjoint appearance references.

This is a bounded engineering hypothesis, not EIG or an identity probability.
Non-goal bearings exclude an already observed side; they do not prove where the
hidden goal is. Counterexample: both objects can be behind the same edge.
"""
import math
import numpy as np
from task_search_actions_20261014 import LiveTaskSearchControl
from live_clear_reference_budget_controls_20261013 import LiveClearReferenceControl as FrozenControl
from visibility_search_actions_20261011 import right_motion,coverage_pointing,search_yaw_target
from causal_identity_view_v4 import rotation,pixel_ray

METHODS=('fixed_orbit_left','fixed_orbit_right','all_candidate_away_route','non_goal_away_route')

class LiveIdentityRouteControl(LiveTaskSearchControl):
    def __init__(self,url,token,scale,mode,origin,infer=None):
        if mode not in METHODS:raise ValueError('Unregistered translation pilot')
        super().__init__(url,token,scale,'fixed_orbit_left' if mode.startswith('fixed_orbit_') else 'identity_evidence_search_left',origin,infer)
        self.mode=mode;self.direction=-1.
        self.fixed=FrozenControl(url,token,scale,mode,origin,infer) if mode.startswith('fixed_orbit_') else None

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
        # Both adaptive arms share exactly the preceding pilot's pointing rule.
        if compatible:
            yaw,pitch=coverage_pointing([candidates[selected]['xyxy']],pose,640,360,90.)
            phase='current_goal_candidate_pointing'
        else:
            target=self.origin['yaw']+search_yaw_target(index)
            yaw=float(np.clip((target-pose['yaw']+180.)%360.-180.,-2.,2.));pitch=0.;phase='continue_task_search'
        if self.mode=='all_candidate_away_route':used=list(range(len(candidates)))
        else:used=[selected] if selected is not None and score is not None and score<=0. else []
        bearings=[]
        for i in used:
            b=candidates[i]['xyxy'];ray=rotation(pose)@pixel_ray((b[0]+b[2])/2.,(b[1]+b[3])/2.,640,360,90.)
            bearings.append((math.degrees(math.atan2(ray[1],ray[0]))-self.origin['yaw']+180.)%360.-180.)
        old=self.direction
        if bearings:
            bearing=float(np.mean(bearings))
            if bearing!=0.:self.direction=-1. if bearing>0. else 1.
        move=right_motion(self.origin)*self.direction
        branch=dict(branch=self.mode,phase=phase,pointing_compatible=compatible,appearance_difference=score,
            readout_selected_index=selected,pointing_candidate_indices=[selected] if compatible else [],
            translation_candidate_indices=used,translation_bearings_from_origin_deg=bearings,
            translation_direction=self.direction,translation_direction_changed=self.direction!=old,
            translation_identity_conditioned=self.mode=='non_goal_away_route',identity_state_consumed=True,
            candidate_observation_consumed=True,mask_readout_changed=False,
            hidden_goal_location_inferred=False,score_is_information_gain=False,identity_score_is_probability=False,
            visible_candidates_do_not_imply_goal_found=bool(candidates) and not compatible)
        return dict(x=float(move[0]),y=float(move[1]),z=0.,yaw=yaw,pitch=pitch),branch
