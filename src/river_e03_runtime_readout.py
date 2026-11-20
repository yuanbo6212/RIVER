"""One shared runtime readout; preserve the frozen strong cheap baseline.

The matched-geometry cheap classifier remains a diagnostic. Runtime cheap is
the ORIGINAL HSV auxiliary representative, not a weaker replacement chosen to
increase the apparent benefit of deep computation. No fitting happens here.
"""
from dataclasses import replace
import json

import numpy as np

from river_e02_control import load_development_representative,MODEL_FILE_SHA256
from river_e03_identity_readout import CausalModeReadout,BASE


class SharedIdentityReadout:
    def __init__(self,bundle_path,*,allow_development_use=False):
        self.deep_readout=CausalModeReadout(bundle_path,allow_development_use=allow_development_use)
        self.cheap_model=load_development_representative(BASE/'artifacts/e01_designated_set_association_v2/model.json')

    def register_reference(self,snapshot):
        self.deep_readout.register_reference(snapshot)

    def predict(self,snapshot,mode,deep_output=None):
        if mode=='deep':
            output=self.deep_readout.predict(snapshot,mode,deep_output)
            output['runtime_mode']='deep_joint_geometry_score_representative'
            return output
        if mode!='cheap' or deep_output is not None:
            raise ValueError('cheap carries no uncomputed deep evidence; modes cheap/deep only')
        try:
            if self.deep_readout.bootstrap is None or snapshot.frame.capture_mono_ns < self.deep_readout.bootstrap.frame.capture_mono_ns:
                raise ValueError('registered causal reference required')
            detail=json.loads(snapshot.identity_json)
            if detail.get('gt_inputs') or detail.get('future_frames_available'):
                raise ValueError('noncausal identity payload')
            lookup={s['index']:s for s in detail['candidate_scores']}
            if set(lookup)!=set(range(len(snapshot.candidates))):
                raise ValueError('unscored current proposal')
            if snapshot.candidates and detail['negative_reference_count']==0:
                raise ValueError('missing frozen negative reference')
            scores=np.array([[lookup[i]['positive_similarity'],lookup[i]['known_negative_similarity']]
                             for i in range(len(snapshot.candidates))],float).reshape(-1,2)
            p=self.cheap_model.predict(scores)
            supported=True;reason='unchanged_frozen_HSV_representative'
        except ValueError as error:
            p=np.zeros(len(snapshot.candidates)+1);p[0]=1.
            scores=None;supported=False;reason='unsupported_current_observation: '+str(error)
        return dict(frame_id=snapshot.frame.frame_id,source_step=snapshot.frame.source_step,
            capture_mono_ns=snapshot.frame.capture_mono_ns,jpeg_sha256=snapshot.frame.jpeg_sha256,
            candidate_sha256=snapshot.candidate_sha256,mode='cheap',probabilities=p.tolist(),
            selected_index=int(np.argmax(p))-1 if int(np.argmax(p)) else None,
            supported=supported,reason=reason,runtime_mode='original_HSV_representative_v2',
            model_sha256=MODEL_FILE_SHA256,probability_scope='auxiliary_uniform_representative_not_physical_state',
            fusion='same_capture_deep_replaces_cheap; never multiply',gt_input=False)

    @staticmethod
    def selected_snapshot(snapshot,readout):
        original=json.loads(snapshot.identity_json)
        future=original.get('future_frames_available')
        if original.get('gt_inputs') is not False or not (future is False or type(future) is int and future==0):
            raise ValueError('cannot sanitize a noncausal original identity payload')
        if any(readout.get(k)!=v for k,v in [
                ('frame_id',snapshot.frame.frame_id),('source_step',snapshot.frame.source_step),
                ('capture_mono_ns',snapshot.frame.capture_mono_ns),('jpeg_sha256',snapshot.frame.jpeg_sha256),
                ('candidate_sha256',snapshot.candidate_sha256)]):
            raise ValueError('readout belongs to another capture')
        probabilities=np.asarray(readout['probabilities'],float)
        if (probabilities.shape!=(len(snapshot.candidates)+1,) or not np.isfinite(probabilities).all() or
                np.any(probabilities<0) or not np.isclose(probabilities.sum(),1.,atol=1e-10,rtol=0.)):
            raise ValueError('invalid source distribution')
        selected=int(np.argmax(probabilities))-1 if int(np.argmax(probabilities)) else None
        if readout['selected_index']!=selected:raise ValueError('readout selection mismatch')
        identity=dict(original,selected_index=selected,
            runtime_mode_readout=readout,gt_inputs=False,future_frames_available=0)
        return replace(snapshot,identity_json=json.dumps(identity,sort_keys=True,allow_nan=False))
