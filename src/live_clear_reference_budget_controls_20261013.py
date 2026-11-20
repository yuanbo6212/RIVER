"""Transparent strong controls, preserving the existing context arms exactly.

The old nominal arm caps prior displacement at 0.04 m. Add a full-permitted
prior arm and a full-permitted ascent; do not reinterpret the context nominal
as a fully budgeted strong baseline. No method or scene outcome is read.
"""
import base64
import numpy as np
from live_clear_reference_strong_controls_20261013 import LiveClearReferenceControl as ContextControl
from live_visibility_search_policy_20261011 import online_state,first_pose_command,bounded_command

METHODS=('nominal','prior_full_budget','coverage_rise_no_identity','fixed_rise_full_budget',
         'fixed_orbit_right','fixed_orbit_left','causal_visibility','causal_search')
FULL=('prior_full_budget','fixed_rise_full_budget')
STEP_CAP=3./29.

class LiveClearReferenceControl(ContextControl):
    def __init__(self,url,token,scale,mode,origin,infer=None):
        if mode not in METHODS:raise ValueError('Unregistered strong control')
        super().__init__(url,token,scale,'nominal' if mode in FULL else mode,origin,infer)
        self.mode=mode

    def observe(self,jpeg,step,pose,initial_box,candidates,selected_index,identity):
        if self.mode not in FULL:
            return super().observe(jpeg,step,pose,initial_box,candidates,selected_index,identity)
        if not 0<=step<230 or step!=len(self.history) or ((step==0)!=(initial_box is not None)):
            raise ValueError('Unordered observation or later reference')
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
        if chunk.shape!=(8,5) or not np.isfinite(chunk).all():raise ValueError('Invalid frozen prior')
        prior=first_pose_command(chunk,self.scale,pose,dt=.1)
        delta=np.array([prior[k]-pose[k] for k in ('x','y','z')]);length=float(np.linalg.norm(delta))
        if length>STEP_CAP:delta*=STEP_CAP/length
        capped_prior=delta.copy()
        if self.mode=='fixed_rise_full_budget':delta=np.array([0.,0.,STEP_CAP])
        proposal=dict(pose)
        for k,d in zip(('x','y','z'),delta):proposal[k]+=float(d)
        for k,cap in (('yaw',1.),('pitch',.5)):
            proposal[k]+=float(np.clip((prior[k]-pose[k]+180.)%360.-180.,-cap,cap))
        if self.mode=='fixed_rise_full_budget':proposal['pitch']-=.5
        command,cost=bounded_command(pose,proposal,self.origin,max(0.,3.-self.path_length),max_step=STEP_CAP)
        self.path_length+=cost;self.previous_pose=dict(pose)
        branch=dict(branch=self.mode,identity_state_consumed=False,candidate_observation_consumed=False,
            prior_translation_consumed=self.mode=='prior_full_budget',prior_rotation_consumed=True,
            full_permitted_step_translation=True)
        return command,dict(reason=self.mode,rgb_prefix=step+1,seed=seed,pi05_chunk=chunk.tolist(),
            history_steps_latest_first=list(range(step,step-5,-1)),state=state.tolist(),command=command,
            control_residual={k:float(v) for k,v in zip(('x','y','z'),delta-capped_prior)},branch=branch,
            command_path_length_m=self.path_length,readout_selected_index=selected_index,identity_state_step=step,
            task_box_source_step=0,gt_inputs=False,prior_translation_cap_m=STEP_CAP,command_translation_cap_m=STEP_CAP,
            score_is_information_gain=False,execute_at_next_observation=True)
