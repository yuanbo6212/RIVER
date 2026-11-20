"""Goal-compatible evidence affects sensing, not the shared mask readout.

Finite strong-control pilot; no calibrated identity probability or EIG claim.
All arms use the same fixed leftward motion and budget. Appearance exclusion
changes when to keep searching and which current observed box to center.
"""
import base64
import numpy as np
from live_visibility_search_policy_20261011 import LiveCoveragePolicy,online_state,first_pose_command,bounded_command
from live_clear_reference_budget_controls_20261013 import LiveClearReferenceControl as FrozenControl
from visibility_search_actions_20261011 import right_motion,coverage_pointing,search_yaw_target

METHODS=('fixed_orbit_left','left_search_any_vehicle','target_servo_no_negative','identity_evidence_search_left')

class LiveTaskSearchControl(LiveCoveragePolicy):
    def __init__(self,url,token,scale,mode,origin,infer=None):
        if mode not in METHODS:raise ValueError('Unregistered finite pilot arm')
        super().__init__(url,token,scale,'fixed_orbit_right',origin,640,360,90.,infer)
        self.mode=mode
        self.fixed=FrozenControl(url,token,scale,'fixed_orbit_left',origin,infer) if mode=='fixed_orbit_left' else None

    def residual(self,index,pose,candidates,selected,identity):
        for c in candidates:
            if set(c)-{'xyxy','confidence','class_id','mask'}:raise ValueError('Non-observation candidate input')
        boxes=[c['xyxy'] for c in candidates]
        evidence=next((x['positive_minus_negative'] for x in identity['candidate_scores'] if x['index']==selected),None)
        if evidence is not None and not np.isfinite(evidence):raise ValueError('Invalid identity evidence')
        if selected is not None and not 0<=selected<len(candidates):raise ValueError('Identity/candidate mismatch')
        if self.mode=='left_search_any_vehicle':point=boxes
        elif self.mode=='target_servo_no_negative':point=[boxes[selected]] if selected is not None else []
        else:point=[boxes[selected]] if selected is not None and evidence is not None and evidence>0. else []
        branch=dict(branch=self.mode,candidate_observation_consumed=True,
            identity_state_consumed=self.mode!='left_search_any_vehicle',negative_evidence_consumed=self.mode=='identity_evidence_search_left',
            mask_readout_changed=False,readout_selected_index=selected,appearance_difference=evidence,
            pointing_candidate_indices=list(range(len(boxes))) if self.mode=='left_search_any_vehicle' and point else ([selected] if point else []),
            score_is_information_gain=False,identity_score_is_probability=False)
        if point:
            yaw,pitch=coverage_pointing(point,pose,640,360,90.)
            branch['phase']='all_current_vehicle_coverage' if self.mode=='left_search_any_vehicle' else 'current_goal_candidate_pointing'
        else:
            target=self.origin['yaw']+search_yaw_target(index)
            yaw=float(np.clip((target-pose['yaw']+180.)%360.-180.,-2.,2.));pitch=0.
            branch.update(phase='continue_task_search',desired_yaw=target,
                visible_candidates_do_not_imply_goal_found=bool(boxes))
        move=-right_motion(self.origin)
        return dict(x=float(move[0]),y=float(move[1]),z=0.,yaw=yaw,pitch=pitch),branch

    def observe(self,jpeg,step,pose,initial_box,candidates,selected_index,identity):
        if self.fixed is not None:
            result=self.fixed.observe(jpeg,step,pose,initial_box,candidates,selected_index,identity)
            self.path_length=self.fixed.path_length
            return result
        if not 0<=step<230 or step!=len(self.history) or ((step==0)!=(initial_box is not None)):
            raise ValueError('Unordered RGB or later reference')
        if identity['step']!=step or identity['selected_index']!=selected_index or identity['gt_inputs'] or identity['future_frames_available']:
            raise ValueError('Noncausal identity ledger')
        self.history.append(jpeg)
        if step==0:self.reference=jpeg;self.initial_box=list(initial_box)
        if not 185<=step<214:
            self.previous_pose=dict(pose)
            return None,dict(reason='shared_fixed_camera_phase' if step<185 else 'fixed_deadline_observation',rgb_prefix=step+1,gt_inputs=False)
        state=online_state(self.previous_pose,pose,self.scale);seed=2026101300+step
        payload=dict(images_b64=[base64.b64encode(x).decode('ascii') for x in reversed(self.history[-5:])],
            reference_b64=base64.b64encode(self.reference).decode('ascii'),reference_box_xyxy=self.initial_box,state=state.tolist(),seed=seed)
        chunk=np.asarray(self.request_prior(payload)['normalized_actions'],float)
        if chunk.shape!=(8,5) or not np.isfinite(chunk).all():raise ValueError('Invalid shared prior')
        prior=first_pose_command(chunk,self.scale,pose,dt=.1)
        delta=np.array([prior[k]-pose[k] for k in ('x','y','z')]);length=float(np.linalg.norm(delta))
        if length>.04:delta*=.04/length
        proposal=dict(pose)
        for k,d in zip(('x','y','z'),delta):proposal[k]+=float(d)
        for k,cap in (('yaw',1.),('pitch',.5)):
            proposal[k]+=float(np.clip((prior[k]-pose[k]+180.)%360.-180.,-cap,cap))
        move,branch=self.residual(step-185,pose,candidates,selected_index,identity)
        for k,v in move.items():proposal[k]+=v
        command,cost=bounded_command(pose,proposal,self.origin,max(0.,3.-self.path_length),max_step=3./29.)
        self.path_length+=cost;self.previous_pose=dict(pose)
        return command,dict(reason=self.mode,rgb_prefix=step+1,seed=seed,pi05_chunk=chunk.tolist(),state=state.tolist(),command=command,
            history_steps_latest_first=list(range(step,step-5,-1)),control_residual=move,branch=branch,
            command_path_length_m=self.path_length,readout_selected_index=selected_index,identity_state_step=step,
            task_box_source_step=0,gt_inputs=False,prior_translation_cap_m=.04,command_translation_cap_m=3./29.,
            score_is_information_gain=False,execute_at_next_observation=True)
