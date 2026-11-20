from pathlib import Path
import json
import math
import sys
import unittest
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'src'))
from river_e03_shared_actions import (ActionLimits,CurrentPointing,ActionIntent,
    CommandMailbox,SharedActionExecutor)
from river_e03_perception import FramePacket,Candidate,LightSnapshot

ORIGIN=dict(x=0.,y=0.,z=5.,pitch=0.,yaw=0.,roll=0.)


def pointing(k,ray=(1.,0.,0.),*,expires=None):
    return CurrentPointing(str(k),str(k),k*.05,k*50_000_000,k*50_000_000,
        (k+2)*50_000_000 if expires is None else expires,tuple(ray),(f'evidence-{k}',))


def step(executor,k,pose,*,lateral='Right',fresh=True,ray=(1.,0.,0.)):
    now=k*50_000_000
    if fresh:
        executor.mailbox.submit(ActionIntent(str(k),lateral,now,k*.05,now+100_000_000,pointing(k,ray)))
    command,issued=executor.tick_start(current_frame_id=str(k),sim_s=k*.05,mono_ns=now,measured_pose=pose)
    effect=executor.post_tick(frame_id=str(k+1),sim_s=(k+1)*.05,mono_ns=now+50_000_000,measured_pose=command)
    return command,issued,effect


class SharedActionsTests(unittest.TestCase):
    def test_left_right_mirror_and_stay_shared_pointing(self):
        arms={name:SharedActionExecutor(ORIGIN) for name in ('Left','Stay','Right')}
        poses={name:dict(ORIGIN) for name in arms}
        ray=np.array([1.,.2,.1]);ray/=np.linalg.norm(ray)
        for k in range(20):
            for name,executor in arms.items():poses[name],_,_=step(executor,k,poses[name],lateral=name,ray=ray)
            self.assertAlmostEqual(poses['Left']['y'],-poses['Right']['y'])
            self.assertEqual(poses['Stay']['y'],0.)
            for key in ('pitch','yaw'):
                self.assertAlmostEqual(poses['Left'][key],poses['Right'][key])
                self.assertAlmostEqual(poses['Left'][key],poses['Stay'][key])

    def test_realized_path_speed_acceleration_and_angular_caps(self):
        executor=SharedActionExecutor(ORIGIN);pose=dict(ORIGIN);previous=0.
        ray=np.array([1.,1.,1.]);ray/=np.linalg.norm(ray)
        for k in range(240):
            pose,issued,effect=step(executor,k,pose,ray=ray)
            velocity=effect['actual_velocity_m_s']
            self.assertLessEqual(abs(velocity),.8+1e-9)
            self.assertLessEqual(abs(velocity-previous)/.05,1.6+1e-7);previous=velocity
            self.assertLessEqual(abs(pose['yaw']),15.+1e-8)
            self.assertLessEqual(abs(pose['pitch']),10.+1e-8)
            self.assertFalse(issued['prior_model_called'])
        self.assertLessEqual(executor.path_m,3.+1e-9)
        self.assertGreater(executor.path_m,2.99)
        self.assertAlmostEqual(executor.path_m,sum(r['actual_step_m'] for r in executor.audit))
        self.assertAlmostEqual(executor.velocity,0.,places=8)

    def test_bounded_old_observation_allowed_then_aged_out_without_restamping(self):
        executor=SharedActionExecutor(ORIGIN)
        executor.mailbox.submit(ActionIntent('observed-direction','Right',0,0.,1_000_000_000,
            pointing(0,expires=1_000_000_000)))
        pose=dict(ORIGIN)
        for k in range(5):
            pose,issued,effect=step(executor,k,pose,fresh=False)
            self.assertIsNone(issued['fallback_reason'])
            self.assertAlmostEqual(issued['reference_age_sim_s'],k*.05)
            self.assertEqual(issued['evidence_source_frame_id'],'0')
        previous=effect['actual_velocity_m_s']
        pose,issued,effect=step(executor,5,pose,fresh=False)
        self.assertEqual(issued['fallback_reason'],'evidence_age_exceeded')
        self.assertEqual(issued['evidence_ids'],[])
        self.assertAlmostEqual(issued['received_reference']['reference_age_sim_s'],.25)
        self.assertFalse(issued['received_reference']['consumed'])
        self.assertAlmostEqual(effect['actual_velocity_m_s'],previous-.08)

    def test_observed_world_bearing_repoints_from_actual_current_camera_pose(self):
        executor=SharedActionExecutor(ORIGIN)
        reference=pointing(0,expires=100_000_000)
        executor.mailbox.submit(ActionIntent('ref','Stay',0,0.,100_000_000,reference))
        measured=dict(ORIGIN,yaw=10.)
        command,row=executor.tick_start(current_frame_id='1',sim_s=.05,mono_ns=50_000_000,measured_pose=measured)
        self.assertLess(command['yaw'],10.)
        self.assertAlmostEqual(row['reference_age_sim_s'],.05)
        self.assertEqual(row['evidence_source_frame_id'],'0')

    def test_future_command_and_future_evidence_rejected(self):
        executor=SharedActionExecutor(ORIGIN)
        executor.mailbox.submit(ActionIntent('future','Right',1,0.,100,None))
        requested,issued=executor.tick_start(current_frame_id='0',sim_s=0.,mono_ns=0,measured_pose=ORIGIN)
        self.assertEqual(issued['fallback_reason'],'future_command')
        executor.post_tick(frame_id='1',sim_s=.05,mono_ns=50_000_000,measured_pose=requested)
        executor.mailbox.submit(ActionIntent('future-evidence','Right',50_000_000,.05,200_000_000,pointing(2)))
        requested,issued=executor.tick_start(current_frame_id='1',sim_s=.05,mono_ns=50_000_000,measured_pose=requested)
        self.assertEqual(issued['fallback_reason'],'future_evidence')
        self.assertIsNone(executor._active)

    def test_expired_command_does_not_extend_deadline(self):
        executor=SharedActionExecutor(ORIGIN)
        executor.mailbox.submit(ActionIntent('expired','Right',0,0.,10,pointing(0)))
        _,issued=executor.tick_start(current_frame_id='0',sim_s=0.,mono_ns=11,measured_pose=ORIGIN)
        self.assertEqual(issued['fallback_reason'],'intent_expired')
        self.assertEqual(issued['intent_expires_mono_ns'],10)
        self.assertEqual(issued['requested_pose'],ORIGIN)

    def test_mailbox_latest_one_slot_and_owner_never_waits(self):
        mailbox=CommandMailbox()
        first=ActionIntent('one','Stay',0,0.,100,None)
        second=ActionIntent('two','Stay',0,0.,100,None)
        self.assertTrue(mailbox.submit(first)['admitted'])
        self.assertEqual(mailbox.submit(second)['replaced_command_id'],'one')
        self.assertEqual(mailbox.poll(),second)
        mailbox._lock.acquire()
        try:
            self.assertIsNone(mailbox.poll())
            self.assertFalse(mailbox.submit(first)['admitted'])
        finally:mailbox._lock.release()

    def test_readback_violation_is_retained_and_stops_executor(self):
        executor=SharedActionExecutor(ORIGIN)
        requested,_=executor.tick_start(current_frame_id='0',sim_s=0.,mono_ns=0,measured_pose=ORIGIN)
        requested['y']+=.1
        with self.assertRaisesRegex(RuntimeError,'readback_invalid'):
            executor.post_tick(frame_id='1',sim_s=.05,mono_ns=50_000_000,measured_pose=requested)
        self.assertTrue(executor.failed)
        self.assertIn('position_readback_mismatch',executor.audit[-1]['violations'])
        with self.assertRaises(ValueError):
            executor.tick_start(current_frame_id='1',sim_s=.05,mono_ns=50_000_000,measured_pose=requested)

    def test_actual_K_light_box_world_bearing_and_unknown(self):
        frame=FramePacket('10',5,1.,1_000_000_000,b'technical-jpeg',640,360,
            (0.,0.,5.,0.,0.,0.),(480.,0.,320.,0.,480.,180.,0.,0.,1.))
        candidate=Candidate((368.,170.,388.,190.),.8,2,b'fixture-only-mask')
        snapshot=LightSnapshot(frame,(candidate,),3,(4,),json.dumps(dict(selected_index=0,gt_inputs=False,future_frames_available=False)))
        result=CurrentPointing.from_light(snapshot,delivered_mono_ns=1_001_000_000,expires_mono_ns=1_100_000_000)
        expected=np.array([1.,58./480.,0.]);expected/=np.linalg.norm(expected)
        np.testing.assert_allclose(result.world_bearing,expected,atol=1e-12)
        snapshot=LightSnapshot(frame,(candidate,),3,(4,),json.dumps(dict(selected_index=None,gt_inputs=False,future_frames_available=False)))
        self.assertIsNone(CurrentPointing.from_light(snapshot,delivered_mono_ns=1_001_000_000,
            expires_mono_ns=1_100_000_000).world_bearing)

    def test_identity_role_or_gt_cannot_enter_light_pointing(self):
        frame=FramePacket('0',0,0.,0,b'fixture',640,360,(0.,0.,5.,0.,0.,0.),
            (480.,0.,320.,0.,480.,180.,0.,0.,1.))
        snapshot=LightSnapshot(frame,(),0,(),json.dumps(dict(selected_index=None,
            gt_inputs=False,future_frames_available=False,target_px=40)))
        with self.assertRaisesRegex(ValueError,'evaluator-only'):
            CurrentPointing.from_light(snapshot,delivered_mono_ns=0,expires_mono_ns=100)

    def test_original_integer_zero_future_flag_only(self):
        frame=FramePacket('0',0,0.,0,b'fixture',640,360,(0.,0.,5.,0.,0.,0.),
            (480.,0.,320.,0.,480.,180.,0.,0.,1.))
        def convert(value,gt=False):
            snapshot=LightSnapshot(frame,(),0,(),json.dumps(dict(selected_index=None,
                gt_inputs=gt,future_frames_available=value)))
            return CurrentPointing.from_light(snapshot,delivered_mono_ns=0,expires_mono_ns=100)
        self.assertIsNone(convert(0).world_bearing)
        for invalid in (None,'0',0.,True,1):
            with self.assertRaisesRegex(ValueError,'noncausal'):convert(invalid)
        for invalid_gt in (None,'False',0,True,1):
            with self.assertRaisesRegex(ValueError,'noncausal'):convert(0,gt=invalid_gt)

    def test_unknown_predeclared_search_moves_but_default_tracking_stays(self):
        for basis,moves in (('identity_conditioned',False),('predeclared_search',True)):
            executor=SharedActionExecutor(ORIGIN)
            executor.mailbox.submit(ActionIntent('search','Right',0,0.,100_000_000,None,motion_basis=basis))
            requested,row=executor.tick_start(current_frame_id='0',sim_s=0.,mono_ns=0,measured_pose=ORIGIN)
            self.assertEqual(requested['y']>0,moves)
            self.assertEqual(row['motion_authorized'],moves)
            self.assertEqual(row['fallback_reason'],'identity_unknown')
            self.assertEqual(requested['yaw'],ORIGIN['yaw'])
            self.assertEqual(row['evidence_ids'],[])

    def test_search_cannot_bypass_expired_future_or_invalid_pointing(self):
        cases=[
            (ActionIntent('expired','Right',0,0.,0,None,motion_basis='predeclared_search'),1,'intent_expired'),
            (ActionIntent('future','Right',1,0.,100,None,motion_basis='predeclared_search'),0,'future_command'),
            (ActionIntent('bad-evidence','Right',0,0.,100,pointing(2),motion_basis='predeclared_search'),0,'future_evidence'),
        ]
        for intent,now,reason in cases:
            executor=SharedActionExecutor(ORIGIN);executor.mailbox.submit(intent)
            requested,row=executor.tick_start(current_frame_id='0',sim_s=0.,mono_ns=now,measured_pose=ORIGIN)
            self.assertEqual(row['fallback_reason'],reason)
            self.assertFalse(row['motion_authorized'])
            self.assertEqual(requested,ORIGIN)


if __name__=='__main__':unittest.main()
