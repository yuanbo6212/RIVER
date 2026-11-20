"""CPU-only mechanical and worst-bound actuator recovery checks; no GT/model."""
from pathlib import Path
import hashlib
import json
import math
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import river_e03_shared_actions as parent
from river_e03_shared_actions_v2 import (ActionLimits,ActionIntent,CurrentPointing,
    SharedActionExecutor,PARENT_SOURCE_SHA256,_robust_angular_interval)

ORIGIN=dict(x=-44.87842559814453,y=-94.9614028930664,z=5.299999713897705,
    pitch=-15.524110794067383,yaw=-179.9136962890625,roll=0.)
TRACE=ROOT/'artifacts/e03_failure_replay_inputs_v1/shared_action_execution.json'


def ray(yaw,pitch):
    y,p=map(math.radians,(yaw,pitch))
    return (math.cos(p)*math.cos(y),math.cos(p)*math.sin(y),math.sin(p))


def advance(ex,k,pose,*,lateral='Right',bearing=None,known=True,noise=None,quantize=False):
    now=k*50_000_000
    ref=CurrentPointing(str(k),str(k),k*.05,now,now,now+100_000_000,bearing,('cpu-source',)) if known else None
    ex.mailbox.submit(ActionIntent(str(k),lateral,now,k*.05,now+100_000_000,ref,
        motion_basis='identity_conditioned'))
    command,issued=ex.tick_start(current_frame_id=str(k),sim_s=k*.05,mono_ns=now,measured_pose=pose)
    actual=dict(command)
    if noise:
        for key,value in noise.items():actual[key]+=value
    if quantize:
        actual={key:float(np.float32(parent._angle(value) if key in ('yaw','pitch') else value))
                for key,value in actual.items()}
    effect=ex.post_tick(frame_id=str(k+1),sim_s=(k+1)*.05,mono_ns=now+50_000_000,measured_pose=actual)
    return actual,issued,effect


class RobustActionsV2Tests(unittest.TestCase):
    def assert_physical(self,ex):
        previous=0.;angular=dict(yaw=0.,pitch=0.)
        for row in ex.audit:
            self.assertFalse(row['violations'])
            self.assertLessEqual(row['actual_step_m']/.05,.8+1e-8)
            v=row['actual_velocity_m_s']
            self.assertLessEqual(abs(v-previous)/.05,1.6+1e-7);previous=v
            for axis in angular:
                w=row['actual_angular_velocity_deg_s'][axis]
                self.assertLessEqual(abs(w),getattr(ex.limits,axis+'_rate_deg_s')+1e-7)
                self.assertLessEqual(abs(w-angular[axis])/.05,getattr(ex.limits,axis+'_accel_deg_s2')+1e-6)
                self.assertLessEqual(abs(parent._angle(row['readback_pose'][axis]-ex.origin[axis])),
                    getattr(ex.limits,axis+'_offset_deg')+1e-8)
                angular[axis]=w
        self.assertLessEqual(ex.path_m,3.+1e-9)
        self.assertAlmostEqual(ex.path_m,sum(r['actual_step_m'] for r in ex.audit),places=12)

    def test_parent_limits_types_source_and_real_failure_numbers(self):
        self.assertIs(ActionLimits,parent.ActionLimits)
        self.assertIs(ActionIntent,parent.ActionIntent)
        self.assertIs(CurrentPointing,parent.CurrentPointing)
        self.assertEqual(hashlib.sha256((ROOT/'src/river_e03_shared_actions.py').read_bytes()).hexdigest(),PARENT_SOURCE_SHA256)
        raw=TRACE.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),
            '5ad1ef05883228829a28c66046f969ed243b5b3539324ae6692ae827e0ec870a')
        rows=json.loads(raw)['audit'];self.assertEqual(len(rows),32)
        self.assertTrue(all(not r['violations'] for r in rows))
        offset=parent._angle(rows[-1]['readback_pose']['yaw']-rows[0]['start_measured_pose']['yaw'])
        previous=rows[-1]['actual_angular_velocity_deg_s']['yaw']
        upper=parent._speed_for_distance(15.-offset,20.,40.,.05)
        self.assertAlmostEqual(offset,10.500762939453125,places=12)
        self.assertAlmostEqual(previous,19.99908447265625,places=12)
        self.assertAlmostEqual(upper,17.998304578993043,places=10)
        self.assertAlmostEqual(previous-2.-upper,.0007798936632070763,places=10)
        with self.assertRaisesRegex(ValueError,'insufficient reserved braking'):
            parent._slew_velocity(previous,20.,40.,.05,-20.,upper)
        # v2 cannot retroactively declare this old near-boundary state robust.
        lower,cap,slew=_robust_angular_interval(15.+offset,15.-offset,20.,40.,.05,.002)
        with self.assertRaisesRegex(ValueError,'insufficient reserved braking'):
            parent._slew_velocity(previous,20.,slew/.05,.05,lower,cap)

    def test_trace_error_prefix_then_float32_world_pose_boundary(self):
        rows=json.loads(TRACE.read_bytes())['audit'];ex=SharedActionExecutor(ORIGIN,max_action_ticks=240)
        pose=dict(ORIGIN)
        for k in range(240):
            # Reuse actual archived per-effect residuals without claiming to
            # reproduce unavailable model bearings. Then nearest-float32 world.
            noise=None
            if k<len(rows):
                r=rows[k];noise={a:parent._angle(r['readback_pose'][a]-r['requested_pose'][a]) for a in ('yaw','pitch')}
            pose,_,_=advance(ex,k,pose,bearing=ray(ORIGIN['yaw']+35.,ORIGIN['pitch']+25.),noise=noise,quantize=True)
        self.assert_physical(ex)
        self.assertLess(abs(parent._angle(pose['yaw']-ORIGIN['yaw'])),15.)

    def test_full_angular_error_bound_mirrored_boundary_and_reversal(self):
        results=[]
        for sign in (-1.,1.):
            origin=dict(ORIGIN,yaw=0.,pitch=0.)
            ex=SharedActionExecutor(origin,max_action_ticks=461);pose=dict(origin)
            for k in range(461):
                direction=sign if k<180 else -sign
                # Worst outward sign, then alternating signs: accepted full bound.
                error=direction*.002 if k<360 else sign*((-1.)**k)*.002
                pose,_,_=advance(ex,k,pose,lateral='Stay',bearing=ray(direction*45.,direction*35.),
                    noise=dict(yaw=error,pitch=error))
            self.assert_physical(ex);results.append(pose)
        for axis in ('yaw','pitch'):self.assertAlmostEqual(results[0][axis],-results[1][axis],places=6)

    def test_long_path_arbitrary_vector_error_and_exact_integration(self):
        steps=3000;ex=SharedActionExecutor(ORIGIN,max_action_ticks=steps);pose=dict(ORIGIN)
        for k in range(steps):
            direction=ex.right if k%3 else np.array([0.,0.,1.])
            error=direction*.0001
            pose,_,_=advance(ex,k,pose,lateral='Right' if k<1200 else 'Left',
                bearing=ray(ORIGIN['yaw'],ORIGIN['pitch']),noise=dict(zip(('x','y','z'),error)))
        self.assert_physical(ex)
        self.assertGreater(ex.path_m,2.5)
        with self.assertRaisesRegex(ValueError,'horizon exhausted'):
            ex.tick_start(current_frame_id=str(steps),sim_s=steps*.05,mono_ns=steps*50_000_000,measured_pose=pose)

    def test_unknown_and_explicit_ctm_stay_stop_within_twelve_float32_ticks(self):
        for mode in ('unknown','ctm_stay'):
            ex=SharedActionExecutor(ORIGIN,max_action_ticks=90);pose=dict(ORIGIN)
            for k in range(90):
                moving=k<78
                pose,issued,effect=advance(ex,k,pose,lateral='Right' if moving else 'Stay',
                    bearing=ray(ORIGIN['yaw']+35.,ORIGIN['pitch']+25.) if moving else ray(pose['yaw'],pose['pitch']),
                    known=moving or mode=='ctm_stay',quantize=True)
                if not moving:self.assertEqual(issued['intended_lateral'],'Stay')
            self.assert_physical(ex)
            self.assertLessEqual(abs(effect['actual_velocity_m_s']),1e-6)
            for w in effect['actual_angular_velocity_deg_s'].values():self.assertLessEqual(abs(w),1e-6)

    def test_all_arms_share_angle_commands_and_limits(self):
        executors={name:SharedActionExecutor(dict(ORIGIN,yaw=0.),max_action_ticks=90) for name in ('Left','Stay','Right')}
        poses={name:dict(e.origin) for name,e in executors.items()}
        for k in range(90):
            for name,ex in executors.items():poses[name],_,_=advance(ex,k,poses[name],lateral=name,bearing=ray(40.,20.))
            for axis in ('yaw','pitch'):
                self.assertEqual(poses['Left'][axis],poses['Stay'][axis]);self.assertEqual(poses['Right'][axis],poses['Stay'][axis])
            self.assertAlmostEqual(poses['Left']['y']-ORIGIN['y'],-(poses['Right']['y']-ORIGIN['y']),places=10)
        for ex in executors.values():self.assert_physical(ex)

    def test_real_readback_violation_is_retained_without_clipping(self):
        ex=SharedActionExecutor(ORIGIN)
        with self.assertRaisesRegex(RuntimeError,'position_readback_mismatch'):
            advance(ex,0,dict(ORIGIN),known=False,noise=dict(y=.0002))
        self.assertTrue(ex.failed)
        self.assertAlmostEqual(ex.audit[-1]['actual_step_m'],.0002)
        self.assertEqual(ex.audit[-1]['readback_pose']['y'],ORIGIN['y']+.0002)
        self.assertNotEqual(ex.audit[-1]['actual_velocity_m_s'],0.)


if __name__=='__main__':unittest.main()
