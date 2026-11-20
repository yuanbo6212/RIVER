"""Finite controller/causal-adapter contracts; not CARLA performance tests."""
from pathlib import Path
import copy
import json
import math
import sys
import tempfile
import unittest
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'src'))
from river_e02_control import (CausalRepresentativeIdentity,E02BranchControl,
    IDENTITY_MODE,load_development_representative,MODEL_FINGERPRINT)


def pose(yaw=0.):
    return dict(x=10.,y=20.,z=5.,pitch=-15.,yaw=yaw,roll=0.)


def candidate(box=(340.,140.,420.,190.)):
    return dict(xyxy=list(box),confidence=.8,class_id=2,mask=np.ones((360,640),np.uint8))


def identity(step,selected=None):
    return dict(step=step,selected_index=selected,gt_inputs=False,future_frames_available=0,
        mode=IDENTITY_MODE,probability_scope='auxiliary_uniform_representative_not_physical_membership')


class FakePrior:
    def __init__(self):self.payloads=[]
    def __call__(self,payload):
        self.payloads.append(copy.deepcopy(payload))
        return dict(normalized_actions=np.full((8,5),.4).tolist())


def state_builder(previous,current,scale):
    return np.zeros(32)


def zero_decoder(chunk,scale,current,dt):
    np.testing.assert_array_equal(chunk,np.zeros((8,5)))
    return dict(current,dt=dt)


def controller(action='right',infer=None,**kwargs):
    return E02BranchControl('http://unused','not-written',np.ones(5),'e02_branch_'+action,pose(),
        dict(image_size_x='640',image_size_y='360',fov='67.380135'),infer=infer or FakePrior(),
        state_builder=state_builder,pose_decoder=zero_decoder,**kwargs)


def advance(c,until=185,selected=None):
    p=pose()
    for step in range(until):
        command,trace=c.observe(b'frame'+str(step).encode(),step,p,[100,100,150,160] if step==0 else None,
            [candidate()] if selected is not None else [],selected,identity(step,selected))
        if command is not None:p=command
    return p


class Extractor:
    def __init__(self,rows=None,negative_count=1):self.step=-1;self.rows=rows;self.negative_count=negative_count
    def update(self,rgb,candidates,step,pose,initial_box,fov):
        self.step=step
        scores=self.rows if self.rows is not None else [dict(index=j,positive_similarity=.98,known_negative_similarity=.1) for j in range(len(candidates))]
        return (0 if candidates else None),dict(step=step,selected_index=0 if candidates else None,
            candidate_scores=scores,gt_inputs=False,future_frames_available=0,template_updated=False,
            negative_memory_updated=False,negative_reference_count=self.negative_count,template_registered=True,
            reference_source='task_intersected_initial_candidate',reference_box_source_step=0)


def adapter(extractor=None,**kw):
    return CausalRepresentativeIdentity(model_path=BASE/'artifacts/e01_designated_set_association_v2/model.json',
        extractor=extractor or Extractor(),allow_development_use=True,**kw)


class E02ControlTests(unittest.TestCase):
    def test_shared_29_real_prior_calls_zero_execution_and_budgets(self):
        for arm,sign in [('left',-1),('stay',0),('right',1)]:
            prior=FakePrior();c=controller(arm,prior);p=pose()
            calls=[]
            for step in range(230):
                cmd,trace=c.observe(b'jpeg'+str(step).encode(),step,p,[100,100,150,160] if step==0 else None,[],None,identity(step))
                if cmd is not None:
                    calls.append(step);p=cmd
                    self.assertFalse(trace['prior_consumed_by_action'])
                    np.testing.assert_array_equal(trace['pi05_chunk'],np.full((8,5),.4))
                    np.testing.assert_array_equal(trace['action_decode_chunk'],np.zeros((8,5)))
                    self.assertLessEqual(trace['step_command_path_m'],3/29)
                    self.assertLessEqual(abs(p['yaw']-pose()['yaw']),15.)
            self.assertEqual(calls,list(range(185,214)))
            self.assertEqual(len(prior.payloads),29)
            self.assertAlmostEqual(c.path_length,abs(sign)*.08*29)
            self.assertAlmostEqual(p['y']-pose()['y'],sign*.08*29)
            self.assertAlmostEqual(p['x'],pose()['x'])
            first=prior.payloads[0]
            self.assertEqual(first['seed'],2026101485)
            self.assertEqual(len(first['images_b64']),5)
            self.assertEqual(set(first),{'images_b64','reference_b64','reference_box_xyxy','state','seed'})

    def test_pointing_function_same_for_all_three_arms_actual_intrinsics(self):
        traces=[]
        for arm in ['left','stay','right']:
            c=controller(arm);p=advance(c,selected=0)
            cmd,trace=c.observe(b'at185',185,p,None,[candidate()],0,identity(185,0))
            traces.append(trace)
            self.assertAlmostEqual(trace['action_camera_intrinsics']['fx'],480.00000047155703)
        self.assertEqual([t['control_residual']['yaw'] for t in traces],[2.,2.,2.])
        self.assertEqual(traces[0]['control_residual']['pitch'],traces[2]['control_residual']['pitch'])
        self.assertEqual([t['control_residual']['y'] for t in traces],[-.08,0.,.08])

    def test_origin_right_axis_not_global_y(self):
        c=E02BranchControl('u','t',np.ones(5),'e02_branch_right',pose(90.),dict(image_size_x=640,image_size_y=360,fov=67.380135),
            infer=FakePrior(),state_builder=state_builder,pose_decoder=zero_decoder)
        p=pose(90.)
        for s in range(186):
            cmd,tr=c.observe(b'x',s,p,[1,1,11,11] if s==0 else None,[],None,identity(s))
        self.assertAlmostEqual(cmd['x'],p['x']-.08)
        self.assertAlmostEqual(cmd['y'],p['y'])

    def test_late_task_stale_identity_gt_and_stale_native_rejected(self):
        c=controller();advance(c,until=1)
        with self.assertRaises(ValueError):c.observe(b'x',1,pose(),[1,1,4,4],[],None,identity(1))
        for bad in [dict(identity(1),step=0),dict(identity(1),gt_inputs=True),dict(identity(1),exit_mode='same')]:
            with self.assertRaises(ValueError):c.observe(b'x',1,pose(),None,[],None,bad)
        with self.assertRaises(ValueError):c.observe(b'x',1,pose(),None,[],None,identity(1),dict(step=0,delivered_prefix=1))

    def test_hidden_candidate_fields_and_active_prior_prohibited(self):
        with self.assertRaises(ValueError):controller(prior_enabled=True)
        c=controller();bad=candidate();bad['actor_id']=123
        with self.assertRaises(ValueError):c.observe(b'x',0,pose(),[1,1,5,5],[bad],None,identity(0))

    def test_bad_real_model_chunk_stops_trial(self):
        c=controller(infer=lambda q:dict(normalized_actions=[[0.]*5]))
        p=advance(c)
        with self.assertRaises(ValueError):c.observe(b'x',185,p,None,[],None,identity(185))

    def test_stay_has_zero_permitted_translation_not_fake_equal_spend(self):
        c=controller('stay');p=advance(c)
        cmd,trace=c.observe(b'x',185,p,None,[],None,identity(185))
        self.assertEqual(trace['step_command_path_m'],0)
        self.assertEqual(trace['command_path_length_m'],0)
        self.assertEqual([cmd[k] for k in ['x','y','z']],[p[k] for k in ['x','y','z']])

    def test_zero_decode_angular_wrap_equivalence_not_false_motion(self):
        c=controller('stay')
        c.pose_decoder=lambda chunk,scale,current,dt:dict(current,yaw=current['yaw']+360.)
        p=advance(c)
        command,trace=c.observe(b'x',185,p,None,[],None,identity(185))
        self.assertEqual(trace['step_command_path_m'],0.)

    def test_original_zero_roll_decoder_accepts_admitted_sensor_roll_only(self):
        c=controller('stay')
        c.pose_decoder=lambda chunk,scale,current,dt:dict(current,roll=0.)
        p=advance(c);p['roll']=1e-6
        command,trace=c.observe(b'x',185,p,None,[],None,identity(185))
        self.assertEqual(command['roll'],0.)
        self.assertEqual(trace['step_command_path_m'],0.)
        self.assertEqual(p['roll'],1e-6)

    def test_zero_decoder_nonzero_roll_or_translation_still_rejected(self):
        for change in (dict(roll=.01),dict(x=100.)):
            c=controller('stay')
            c.pose_decoder=lambda chunk,scale,current,dt:dict(current,**change)
            p=advance(c)
            with self.assertRaisesRegex(ValueError,'zero-decoded'):
                c.observe(b'x',185,p,None,[],None,identity(185))


class E02IdentityTests(unittest.TestCase):
    def test_frozen_corrected_model_fingerprint_and_no_release_flag_rewrite(self):
        m=load_development_representative(BASE/'artifacts/e01_designated_set_association_v2/model.json')
        self.assertEqual(m.metadata['model_sha256'],MODEL_FINGERPRINT)
        self.assertFalse(m.metadata['policy_release'])
        with self.assertRaises(ValueError):CausalRepresentativeIdentity(extractor=Extractor())

    def test_wrong_frozen_model_or_corrupt_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'model.json'
            a=json.loads((BASE/'artifacts/e01_designated_set_association_v2/model.json').read_text())
            a['metadata']['data_id']='e01_hsv_export_v1/withdrawn'
            p.write_text(json.dumps(a))
            with self.assertRaises(ValueError):load_development_representative(p)

    def test_current_model_choice_and_empty_null(self):
        a=adapter();im=np.zeros((360,640,3),np.uint8)
        self.assertEqual(a.update(im,[candidate()],0,pose(),[340,140,420,190])[0],0)
        selected,detail=a.update(im,[candidate()],1,pose())
        probs=a.model.predict([[.98,.1]])
        self.assertEqual(selected,int(np.argmax(probs))-1 if np.argmax(probs) else None)
        self.assertEqual(detail['auxiliary_probabilities'],probs.tolist())
        self.assertEqual(detail['mode'],IDENTITY_MODE)
        selected,detail=a.update(im,[],2,pose())
        self.assertIsNone(selected)
        self.assertEqual(detail['auxiliary_probabilities'],[1.])

    def test_unscored_candidates_retained_without_fake_probability(self):
        a=adapter(Extractor(rows=[dict(index=0,positive_similarity=.98,known_negative_similarity=.1)]))
        im=np.zeros((360,640,3),np.uint8);two=[candidate(),candidate((20,20,40,40))]
        a.update(im,two,0,pose(),[340,140,420,190])
        selected,detail=a.update(im,two,1,pose())
        self.assertIsNone(selected);self.assertIsNone(detail['auxiliary_probabilities'])
        self.assertEqual(detail['unscored_candidate_count'],1)

    def test_missing_negative_reference_unknown_and_annotation_clock(self):
        a=adapter(Extractor(negative_count=0));im=np.zeros((360,640,3),np.uint8)
        a.update(im,[candidate()],0,pose(),[340,140,420,190])
        with self.assertRaises(ValueError):a.update(im,[candidate()],1,pose(),[340,140,420,190])
        selected,detail=a.update(im,[candidate()],1,pose())
        self.assertIsNone(selected);self.assertIsNone(detail['auxiliary_probabilities'])
        self.assertEqual(detail['reason'],'missing_frozen_negative_reference_unknown')


if __name__=='__main__':unittest.main()
