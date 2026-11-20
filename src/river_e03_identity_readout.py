"""Frozen candidate-score models plus mode-comparable auxiliary association.

No fitting, GT, simulator, network or model backbone is imported here. The
categorical target is an auxiliary uniform representative, not physical state.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from river_calibration import HistogramSpec
from river_logistic import sigmoid
from river_set_association import RepresentativeAssociationModel
from river_e03_perception import LightSnapshot

BASE = Path(__file__).resolve().parents[1]
FEATURES = ('hsv_positive','hsv_negative','dino_positive','dino_negative',
            'deep_negative_reference_present','shape_task_cost','shape_context_cost',
            'context_reference_present','bearing_distance_task_rad','geometry_valid')
MODE_COLUMNS = {'cheap': (0,1,5,6,7,8,9), 'deep': tuple(range(10))}


def require(value, message):
    if not value:
        raise ValueError(message)


def candidate_class_score(parameters, features):
    x = np.asarray(features, float)
    require(x.ndim == 2 and x.shape[1] == 10 and np.isfinite(x).all(), 'finite Nx10 observed features required')
    columns = np.asarray(parameters['columns'], int)
    mean, scale, weights = (np.asarray(parameters[k], float) for k in ('mean','scale','coefficients'))
    require(mean.shape == scale.shape == (len(columns),) and weights.shape == (len(columns)+1,), 'invalid frozen logistic schema')
    require(np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale > 0) and np.isfinite(weights).all(), 'invalid frozen logistic values')
    return sigmoid(weights[0] + ((x[:,columns]-mean)/scale) @ weights[1:])


def representative_from_artifact(a):
    spec = HistogramSpec(tuple(a['feature_names']),tuple(np.array(e) for e in a['bin_edges']),
                         a['histogram_pseudocount'],a['observation_definition'])
    return RepresentativeAssociationModel(spec,a['target_bin_mass'],a['nuisance_bin_mass'],
                                         a['candidate_present_prior_q'],a['expected_nuisance_count'],a['metadata'])


class CausalModeReadout:
    def __init__(self, bundle_path, *, allow_development_use=False):
        require(allow_development_use, 'explicit bounded development use required')
        raw = Path(bundle_path).read_bytes()
        self.bundle_sha256 = hashlib.sha256(raw).hexdigest()
        self.bundle = json.loads(raw)
        require(self.bundle['schema'] == 'river.e03.fixed_mode_representative.v1', 'wrong mode bundle')
        self.models = {}
        for mode, columns in MODE_COLUMNS.items():
            a = self.bundle['modes'][mode]
            require(tuple(a['logistic']['columns']) == columns, 'mode feature columns changed')
            self.models[mode] = representative_from_artifact(a['representative'])
        require(abs(self.models['cheap'].candidate_present_prior_q-self.models['deep'].candidate_present_prior_q) < 1e-12,
                'modes must share proposal exposure q')
        require(abs(self.models['cheap'].expected_nuisance_count-self.models['deep'].expected_nuisance_count) < 1e-12,
                'modes must share nuisance exposure')
        sys.path.insert(0,str(BASE/'source_reference'))
        from causal_identity_view_v4 import rotation, pixel_ray, rotation_normalized_box
        self.rotation,self.pixel_ray,self.normalized_box = rotation,pixel_ray,rotation_normalized_box
        self.bootstrap = None

    def _bearing(self, box, frame):
        pose = dict(zip(('x','y','z','pitch','yaw','roll'),frame.pose))
        fov = math.degrees(2*math.atan(frame.width/(2*frame.intrinsic[0])))
        r = self.rotation(pose) @ self.pixel_ray((box[0]+box[2])/2,(box[1]+box[3])/2,frame.width,frame.height,fov)
        return r/np.linalg.norm(r)

    def register_reference(self, snapshot):
        require(self.bootstrap is None and isinstance(snapshot,LightSnapshot) and
                snapshot.frame.source_step == 0 and snapshot.initial_box is not None, 'one legal live F0 reference required')
        self.bootstrap = snapshot
        box = snapshot.initial_box
        self.task_height = box[3]-box[1]
        self.task_bearing = self._bearing(box,snapshot.frame)
        contexts = [c for c in snapshot.candidates if min(c.xyxy[2],box[2]) <= max(c.xyxy[0],box[0]) or
                    min(c.xyxy[3],box[3]) <= max(c.xyxy[1],box[1])]
        context = max(contexts,key=lambda c:c.confidence) if contexts else None
        self.context_height = None if context is None else context.xyxy[3]-context.xyxy[1]

    def features(self,snapshot,mode,deep_output=None):
        require(mode in MODE_COLUMNS and self.bootstrap is not None and isinstance(snapshot,LightSnapshot), 'registered mode/snapshot required')
        require(snapshot.frame.capture_mono_ns >= self.bootstrap.frame.capture_mono_ns, 'observation predates reference')
        detail = json.loads(snapshot.identity_json)
        require(not detail.get('gt_inputs') and not detail.get('future_frames_available'), 'noncausal identity payload')
        scores = {s['index']:s for s in detail['candidate_scores']}
        require(set(scores) == set(range(len(snapshot.candidates))), 'unscored candidate; do not discard/renormalize subset')
        if snapshot.candidates:
            require(detail['negative_reference_count'] > 0, 'frozen negative-reference domain unavailable')
        deep_scores = {}
        if mode == 'deep':
            require(isinstance(deep_output,dict), 'deep evidence must be computed')
            for key,expected in [('frame_id',snapshot.frame.frame_id),('source_step',snapshot.frame.source_step),
                                 ('capture_mono_ns',snapshot.frame.capture_mono_ns),('jpeg_sha256',snapshot.frame.jpeg_sha256),
                                 ('candidate_sha256',snapshot.candidate_sha256),('candidate_count',len(snapshot.candidates))]:
                require(deep_output.get(key) == expected,'deep capture binding mismatch: '+key)
            deep_scores = {s['candidate_index']:s for s in deep_output['scores']}
            require(len(deep_scores) == len(deep_output['scores']) and set(deep_scores) == set(scores), 'deep candidate coverage mismatch')
        origin = dict(zip(('x','y','z','pitch','yaw','roll'),self.bootstrap.frame.pose))
        pose = dict(zip(('x','y','z','pitch','yaw','roll'),snapshot.frame.pose))
        f = snapshot.frame
        require((f.width,f.height,f.intrinsic) == (self.bootstrap.frame.width,self.bootstrap.frame.height,self.bootstrap.frame.intrinsic),
                'sensor calibration changed')
        fov = math.degrees(2*math.atan(f.width/(2*f.intrinsic[0])))
        rows=[]
        for i,c in enumerate(snapshot.candidates):
            s=scores[i]
            hp,hn = float(s['positive_similarity']),float(s['known_negative_similarity'])
            require(np.isfinite([hp,hn]).all() and -1e-12 <= min(hp,hn) and max(hp,hn) <= 1+1e-12, 'HSV outside original numerical domain')
            norm = self.normalized_box(c.xyxy,pose,origin,f.width,f.height,fov)
            valid = norm is not None and norm[3] > norm[1]
            height = norm[3]-norm[1] if valid else None
            shape = abs(math.log(height/self.task_height)) if valid else 0.
            context = abs(math.log(height/self.context_height)) if valid and self.context_height is not None else 0.
            distance = math.acos(float(np.clip(self._bearing(c.xyxy,f)@self.task_bearing,-1,1))) if valid else 0.
            dp=dn=present=0.
            if mode == 'deep':
                d=deep_scores[i];present=float(bool(d['negative_reference_present']))
                require(present == 1. and d['negative_reference_count'] > 0, 'deep negative reference outside fitted domain')
                dp,dn=float(d['positive_similarity']),float(d['known_negative_similarity'])
                require(np.isfinite([dp,dn]).all() and min(dp,dn) >= -1-1e-6 and max(dp,dn) <= 1+1e-6, 'deep cosine invalid')
            rows.append([np.clip(hp,0,1),np.clip(hn,0,1),dp,dn,present,shape,context,
                         float(self.context_height is not None),distance,float(valid)])
        return np.asarray(rows,float).reshape(-1,10)

    def predict(self,snapshot,mode,deep_output=None):
        try:
            x=self.features(snapshot,mode,deep_output)
            observation=candidate_class_score(self.bundle['modes'][mode]['logistic'],x)
            p=self.models[mode].predict(observation.reshape(-1,1))
            reason='fixed_mode_auxiliary_representative'
            supported=True
        except ValueError as error:
            p=np.zeros(len(snapshot.candidates)+1);p[0]=1.
            observation=None;reason='unsupported_current_observation: '+str(error);supported=False
        return dict(frame_id=snapshot.frame.frame_id,source_step=snapshot.frame.source_step,
            capture_mono_ns=snapshot.frame.capture_mono_ns,jpeg_sha256=snapshot.frame.jpeg_sha256,
            candidate_sha256=snapshot.candidate_sha256,mode=mode,probabilities=p.tolist(),
            selected_index=int(np.argmax(p))-1 if int(np.argmax(p)) else None,
            candidate_class_observation_scores=None if observation is None else observation.tolist(),
            supported=supported,reason=reason,model_bundle_sha256=self.bundle_sha256,
            probability_scope='auxiliary_uniform_representative_not_physical_state',
            fusion='same_capture_deep_replaces_cheap; never multiply',gt_input=False)
