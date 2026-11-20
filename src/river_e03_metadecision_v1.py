"""Task-value macro choice; consumes no uncomputed scene or future outcome."""
import hashlib,json,math
from pathlib import Path
import numpy as np
OPTIONS=('adaptive_left','adaptive_right','scene_feedback')
POLICIES=('J3-independent','J4-task-value','J3-full-scene')
FIELDS=('latest_light_candidate_count','latest_auxiliary_null_scalar','last_accepted_age_sim_s',
    'last_accepted_box_width_fraction','last_accepted_box_height_fraction',
    'last_accepted_bearing_initial_optical_0','last_accepted_bearing_initial_optical_1',
    'accepted_pair_unit_ray_slope_initial_optical_per_s_0','accepted_pair_unit_ray_slope_initial_optical_per_s_1')

def choose(policy,risks,*,target_verified,compute_feasible):
    if policy not in POLICIES or type(target_verified) is not bool:raise ValueError('explicit policy and observed verification required')
    if set(risks)!=set(OPTIONS) or not all(math.isfinite(float(v)) and 0<=v<=5 for v in risks.values()):raise ValueError('finite supported macro risks required')
    cheap=min(OPTIONS[:2],key=lambda a:(risks[a],OPTIONS.index(a)))
    if policy=='J4-task-value':
        eligible=OPTIONS if compute_feasible else OPTIONS[:2]
        selected=min(eligible,key=lambda a:(risks[a],int(a=='scene_feedback'),OPTIONS.index(a)))
    elif compute_feasible and (policy=='J3-full-scene' or not target_verified):selected='scene_feedback'
    else:selected=cheap
    return dict(option=selected,compute=selected=='scene_feedback',cheap_fallback=cheap,
        direction=None if selected=='scene_feedback' else ('Left' if selected=='adaptive_left' else 'Right'),
        cheap_direction='Left' if cheap=='adaptive_left' else 'Right',policy=policy)

class MacroRiskModel:
    def __init__(self,root):
        root=Path(root);receipt=json.loads((root/'freeze_receipt.json').read_text());raw=(root/'model.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=receipt['model_sha256']:raise ValueError('macro model checksum mismatch')
        self.model=json.loads(raw)
        if tuple(self.model['fields'])!=FIELDS:raise ValueError('unapproved macro feature fields')

    def predict(self,features,options=OPTIONS):
        x=np.array([round(float(features[f]),2) for f in FIELDS])
        if not np.isfinite(x).all():raise ValueError('missing/nonfinite causal feature')
        result={}
        for option,models in self.model['heads'].items():
            if option not in options:continue
            values={}
            for name,m in models.items():
                values[name]=float(np.clip(m['intercept']+((x-np.array(m['mean']))/np.array(m['scale']))@np.array(m['weight']),0,1))
            result[option]=dict(**values,risk=1-values['retained']+4*values['wrong'])
        return result

    def select(self,policy,features,*,source_step,capture_ns,available_ns,decision_ns,deadline_ns,target_verified,remaining_calls=1):
        if source_step!=180 or not 0<=capture_ns<=available_ns<=decision_ns<deadline_ns:raise ValueError('causal source180 decision window required')
        if policy not in POLICIES or type(target_verified) is not bool:raise ValueError('explicit policy/status required')
        feasible=remaining_calls>=1
        # Independent baselines do not pay for a task-value forecast they do
        # not need. A forced scene decision contains no fabricated risk scores.
        if feasible and (policy=='J3-full-scene' or policy=='J3-independent' and not target_verified):
            return dict(option='scene_feedback',compute=True,direction=None,policy=policy,predictions=None,
                GT_inputs=False,uncomputed_scene_inputs=False,value_scope='independent tracking-status trigger')
        if policy!='J4-task-value':
            values=self.predict(features,OPTIONS[:2]);option=min(OPTIONS[:2],key=lambda a:(values[a]['risk'],OPTIONS.index(a)))
            return dict(option=option,compute=False,direction='Left' if option=='adaptive_left' else 'Right',policy=policy,
                predictions=values,GT_inputs=False,uncomputed_scene_inputs=False,value_scope='shared cheap-action risk predictor')
        values=self.predict(features)
        result=choose(policy,{k:v['risk'] for k,v in values.items()},target_verified=target_verified,
            compute_feasible=feasible)
        return dict(result,predictions=values,GT_inputs=False,uncomputed_scene_inputs=False,
            value_scope='fixed conditional task-outcome regressions, not physical posterior')
