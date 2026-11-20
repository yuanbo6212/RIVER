"""Shared orbit translation, causal task/pair pointing: finite development controls.

These are transparent strong control baselines, not calibrated information gain.
Each mode shares exactly the same ray/plane readout and frozen pi0.5 proposal.
"""
import base64
import json
import math
import sys
from pathlib import Path
from urllib.request import Request,urlopen
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'cosfly-pi05/src'))
from cosfly_pi05.control import first_pose_command,online_state
from live_ck6_policy import bounded_command
from calibrated_view_actions_v5 import focal_length

ACTION_STEPS=tuple(range(165,194))
from visibility_search_actions_20261011 import MODES, action_residual


class LiveCoveragePolicy:
    def __init__(self,url,token,scale,mode,origin,width,height,fov,infer=None):
        focal_length(width,height,fov)
        if mode not in MODES:raise ValueError('unfrozen pointing mode')
        self.intrinsics=dict(width=width,height=height,fov=fov)
        self.width,self.height,self.fov=width,height,fov
        self.url,self.token,self.scale,self.mode=url.rstrip('/'),token,np.asarray(scale,float),mode
        self.origin=dict(origin);self.history=[];self.reference=None;self.initial_box=None
        self.previous_pose=None;self.path_length=0.;self.infer=infer

    def request_prior(self,payload):
        if self.infer is not None:return self.infer(payload)
        request=Request(self.url+'/infer',data=json.dumps(payload).encode('utf-8'),
                        headers={'Content-Type':'application/json','Authorization':'Bearer '+self.token})
        with urlopen(request,timeout=180) as response:return json.load(response)

    def observe(self,jpeg,step,pose,initial_box=None,candidates=(),selected_index=None,identity=None):
        if step!=len(self.history) or ((step==0)!=(initial_box is not None)):
            raise ValueError('unordered RGB or later task annotation')
        if identity is None or identity['step']!=step or identity['selected_index']!=selected_index:
            raise ValueError('action/readout mismatch')
        if identity['gt_inputs'] or identity['future_frames_available']!=0:raise ValueError('noncausal state')
        self.history.append(jpeg)
        if step==0:self.reference=jpeg;self.initial_box=list(initial_box)
        if step not in ACTION_STEPS:
            self.previous_pose=dict(pose)
            return None,dict(reason='shared_fixed_camera_phase' if step<165 else 'fixed_deadline_observation',rgb_prefix=step+1,gt_inputs=False)
        frames=list(reversed(self.history[-5:]));state=online_state(self.previous_pose,pose,self.scale)
        seed=2026100800+step
        payload=dict(images_b64=[base64.b64encode(x).decode('ascii') for x in frames],
            reference_b64=base64.b64encode(self.reference).decode('ascii'),reference_box_xyxy=self.initial_box,state=state.tolist(),seed=seed)
        chunk=np.asarray(self.request_prior(payload)['normalized_actions'],float)
        if chunk.shape!=(8,5) or not np.isfinite(chunk).all():raise ValueError('invalid frozen prior')
        prior=first_pose_command(chunk,self.scale,pose,dt=.1)
        delta=np.array([prior[k]-pose[k] for k in ('x','y','z')]);length=float(np.linalg.norm(delta))
        if length>.04:delta*=.04/length
        proposal=dict(pose)
        for k,d in zip(('x','y','z'),delta):proposal[k]+=float(d)
        for k,cap in (('yaw',1.),('pitch',.5)):
            change=(prior[k]-pose[k]+180)%360-180;proposal[k]+=float(np.clip(change,-cap,cap))
        residual,branch=action_residual(self.mode,step-165,candidates,selected_index,identity,pose,self.origin,self.width,self.height,self.fov)
        for k,v in residual.items():proposal[k]+=v
        command,cost=bounded_command(pose,proposal,self.origin,max(0.,3.-self.path_length),max_step=3./29.)
        self.path_length+=cost
        record=dict(reason=self.mode,rgb_prefix=step+1,seed=seed,pi05_chunk=chunk.tolist(),
            history_steps_latest_first=list(range(step,step-5,-1)),state=state.tolist(),command=command,
            command_path_length_m=self.path_length,task_box_source_step=0,gt_inputs=False,control_residual=residual,
            branch=branch,readout_selected_index=selected_index,identity_state_step=identity['step'],
            observer_consumed_by_action=branch.get('candidate_observation_consumed',branch['identity_state_consumed']),prior_translation_cap_m=.04,
            command_translation_cap_m=3./29.,projection_score_is_information_gain=False,action_camera_intrinsics=dict(self.intrinsics),
            readout='ground_range_pair',assumed_initial_camera_height_m=5.)
        self.previous_pose=dict(pose)
        return command,record
