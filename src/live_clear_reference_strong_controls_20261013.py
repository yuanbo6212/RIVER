"""230-step controls, first-frame reference and strictly next-step commands.

Baseline preparation only. No failed identity switch or actor state is read.
"""
import base64,math
import numpy as np
from live_visibility_search_policy_20261011 import LiveCoveragePolicy,online_state,first_pose_command,bounded_command
from visibility_search_actions_20261011 import action_residual,coverage_pointing

METHODS=('nominal','fixed_orbit_right','fixed_rise','coverage_rise_no_identity','causal_visibility','causal_search','fixed_orbit_left')
ACTION_STEPS=tuple(range(185,214))

def residual(mode,index,candidates,selected,identity,pose,origin):
    if mode=='fixed_orbit_left':
        delta,detail=action_residual('fixed_orbit_right',index,candidates,selected,identity,pose,origin,640,360,90.)
        for key in ('x','y','yaw'):delta[key]=-delta[key]
        detail=dict(detail,branch=mode)
        return delta,detail
    if mode in ('fixed_orbit_right','causal_visibility','causal_search'):
        return action_residual(mode,index,candidates,selected,identity,pose,origin,640,360,90.)
    if mode=='nominal':return {},dict(branch=mode,identity_state_consumed=False)
    if mode=='fixed_rise':return dict(x=0.,y=0.,z=.08,yaw=0.,pitch=-.5),dict(branch=mode,identity_state_consumed=False)
    if mode=='coverage_rise_no_identity':
        boxes=[c['xyxy'] for c in candidates]
        yaw,pitch=coverage_pointing(boxes,pose,640,360,90.) if boxes else (0.,-.5)
        return dict(x=0.,y=0.,z=.08,yaw=yaw,pitch=pitch),dict(branch=mode,identity_state_consumed=False,candidate_count=len(boxes))
    raise ValueError('unregistered control')

class LiveClearReferenceControl(LiveCoveragePolicy):
    def __init__(self,url,token,scale,mode,origin,infer=None):
        if mode not in METHODS:raise ValueError('unregistered control')
        super().__init__(url,token,scale,'fixed_orbit_right',origin,640,360,90.,infer)
        self.mode=mode
    def observe(self,jpeg,step,pose,initial_box,candidates,selected_index,identity):
        if not 0<=step<230 or step!=len(self.history) or ((step==0)!=(initial_box is not None)):
            raise ValueError('unordered image or later reference')
        if identity['step']!=step or identity['selected_index']!=selected_index or identity['gt_inputs'] or identity['future_frames_available']:
            raise ValueError('noncausal identity ledger')
        self.history.append(jpeg)
        if step==0:self.reference=jpeg;self.initial_box=list(initial_box)
        if step not in ACTION_STEPS:
            self.previous_pose=dict(pose)
            return None,dict(reason='shared_fixed_camera_phase' if step<185 else 'fixed_deadline_observation',rgb_prefix=step+1,gt_inputs=False)
        state=online_state(self.previous_pose,pose,self.scale);seed=2026101300+step
        payload=dict(images_b64=[base64.b64encode(x).decode('ascii') for x in reversed(self.history[-5:])],
            reference_b64=base64.b64encode(self.reference).decode('ascii'),reference_box_xyxy=self.initial_box,state=state.tolist(),seed=seed)
        chunk=np.asarray(self.request_prior(payload)['normalized_actions'],float)
        if chunk.shape!=(8,5) or not np.isfinite(chunk).all():raise ValueError('invalid frozen prior')
        prior=first_pose_command(chunk,self.scale,pose,dt=.1)
        delta=np.array([prior[k]-pose[k] for k in ('x','y','z')]);length=float(np.linalg.norm(delta))
        if length>.04:delta*=.04/length
        proposal=dict(pose)
        for k,d in zip(('x','y','z'),delta):proposal[k]+=float(d)
        for k,cap in (('yaw',1.),('pitch',.5)):
            proposal[k]+=float(np.clip((prior[k]-pose[k]+180.)%360.-180.,-cap,cap))
        move,branch=residual(self.mode,step-185,candidates,selected_index,identity,pose,self.origin)
        for k,v in move.items():proposal[k]+=v
        command,cost=bounded_command(pose,proposal,self.origin,max(0.,3.-self.path_length),max_step=3./29.)
        self.path_length+=cost;self.previous_pose=dict(pose)
        return command,dict(reason=self.mode,rgb_prefix=step+1,seed=seed,pi05_chunk=chunk.tolist(),
            history_steps_latest_first=list(range(step,step-5,-1)),state=state.tolist(),command=command,
            control_residual=move,branch=branch,command_path_length_m=self.path_length,
            readout_selected_index=selected_index,identity_state_step=step,task_box_source_step=0,
            gt_inputs=False,prior_translation_cap_m=.04,command_translation_cap_m=3./29.,
            score_is_information_gain=False,execute_at_next_observation=True)
