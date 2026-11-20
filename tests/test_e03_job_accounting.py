"""Accounting/binding counterexamples; stamp fixtures are not runtime measurements."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from river_e03_perception import Completion, FramePacket, LightSnapshot, WorkItem
from river_e03_job_accounting import (AccountingError, AdmissionDenied, GpuMeasurement,
                                     JobAccounting, SourceBinding, canonical_schedule)
from river_runtime import ScheduleMode

S = 1_000_000_000


def frame():
    return FramePacket('f0', 0, 0., S, b'explicit-accounting-fixture', 6, 4,
                       (0.,)*6, (3., 0., 3., 0., 3., 2., 0., 0., 1.), 'archived_replay_ingress')


def snapshot():
    return LightSnapshot(frame(), (), 0, (), '{}', (1., 1., 4., 3.))


def item(job='d0', *, expires=20*S):
    return WorkItem(job, snapshot(), expires)


def payload():
    return dict(SourceBinding.from_payload(snapshot()).__dict__, status='ok', scores=[])


def completion(job='d0', *, start=4*S, end=6*S, delivered=7*S, expires=20*S):
    return Completion(job, 'deep', 'f0', S, 3*S, start, end, expires, 'ok', payload(), None, delivered)


def measurement(cuda=.25):
    return GpuMeasurement(cuda_stream_elapsed_s=cuda, cuda_provenance='fixture-independent-stream-meter')


def admit(ledger, work=None, reserve=.5):
    ledger.admit(work or item(), mode='deep', schedule='compute_in_flight',
                 admitted_mono_ns=2*S, reserved_budget_s=reserve)


class JobAccountingTests(unittest.TestCase):
    def test_queue_is_not_compute_and_stream_is_not_gpu_resource(self):
        ledger=JobAccounting(max_deep_budget_s=1.)
        admit(ledger);ledger.mark_started('d0', 4*S)
        result=ledger.settle(completion(), measurement())
        self.assertEqual(result.queue_wall_s, 1.)
        self.assertEqual(result.compute_wall_s, 2.)
        self.assertEqual(result.capture_to_delivery_wall_s, 6.)
        self.assertEqual(result.measurements.cuda_stream_elapsed_s, .25)
        self.assertIsNone(result.measurements.gpu_resource_s)
        self.assertEqual(ledger.summary()['gpu_resource_unknown_job_ids'], ['d0'])
        self.assertAlmostEqual(ledger.summary()['deep_budget_liability_s'], .25)
        self.assertTrue(result.allow_capture_time_update)

    def test_deadline_does_not_refund_running_compute(self):
        ledger=JobAccounting(max_deep_budget_s=1.)
        admit(ledger,item(expires=5*S));ledger.mark_started('d0',4*S)
        ledger.expire_decision('d0',6*S)
        result=ledger.settle(completion(expires=5*S), measurement(.4))
        self.assertFalse(result.allow_capture_time_update)
        self.assertEqual(result.compute_wall_s,2.)
        self.assertAlmostEqual(ledger.summary()['deep_budget_liability_s'],.4)
        with self.assertRaises(AdmissionDenied):admit(ledger,item('d1'))

    def test_unknown_actual_cost_stays_reserved_and_prevents_new_deep(self):
        ledger=JobAccounting(max_deep_jobs=2,max_deep_budget_s=2.)
        admit(ledger)
        result=ledger.settle(completion())
        self.assertTrue(result.measurement_pending)
        self.assertIsNone(result.measurements.cuda_stream_elapsed_s)
        self.assertEqual(ledger.summary()['deep_budget_liability_s'],.5)
        self.assertEqual(ledger.summary()['cuda_unknown_job_ids'],['d0'])
        with self.assertRaises(AdmissionDenied):admit(ledger,item('d1'))

    def test_all_source_bindings_and_timestamp_order_are_enforced(self):
        for key in ('frame_id','source_step','capture_mono_ns','capture_sim_s','jpeg_sha256',
                    'candidate_sha256','candidate_count','capture_pose','capture_intrinsic','capture_origin'):
            with self.subTest(key=key):
                ledger=JobAccounting(max_deep_budget_s=1.);admit(ledger)
                bad=payload();bad[key]='different-source'
                with self.assertRaises(AccountingError):ledger.settle(replace(completion(),payload=bad),measurement())
                self.assertEqual(ledger.summary()['settled_jobs'],0)
                self.assertEqual(ledger.summary()['deep_budget_liability_s'],.5)
        ledger=JobAccounting(max_deep_budget_s=1.);admit(ledger)
        with self.assertRaises(AccountingError):ledger.settle(completion(start=2*S),measurement())
        with self.assertRaises(AccountingError):ledger.settle(replace(completion(),delivered_mono_ns=None),measurement())

    def test_identical_duplicate_is_idempotent_conflict_keeps_original_charge(self):
        ledger=JobAccounting(max_deep_budget_s=1.);admit(ledger)
        first=ledger.settle(completion(),measurement(.3))
        self.assertIs(ledger.settle(completion(),measurement(.3)),first)
        self.assertEqual(ledger.summary()['settled_jobs'],1)
        changed=payload();changed['scores']=[dict(positive_similarity=.9)]
        with self.assertRaises(AccountingError):ledger.settle(replace(completion(),payload=changed),measurement(.3))
        with self.assertRaises(AccountingError):ledger.settle(completion(),measurement(.4))
        self.assertAlmostEqual(ledger.summary()['deep_budget_liability_s'],.3)

    def test_hard_job_limit_reservations_and_actual_overrun(self):
        ledger=JobAccounting(max_deep_jobs=3,max_deep_budget_s=.6);admit(ledger)
        with self.assertRaises(AdmissionDenied):admit(ledger,item('d1'),reserve=.2)
        charged=ledger.settle(completion(),measurement(.8))
        self.assertTrue(charged.reservation_overrun)
        self.assertTrue(charged.budget_overrun)
        self.assertEqual(ledger.summary()['deep_budget_liability_s'],.8)
        with self.assertRaises(AdmissionDenied):admit(ledger,item('d1'),reserve=.1)

    def test_verified_unstarted_expiry_is_distinct_from_timeout(self):
        ledger=JobAccounting(max_deep_budget_s=1.)
        admit(ledger,item(expires=5*S))
        done=completion(start=6*S,end=6*S+100,delivered=7*S,expires=5*S)
        done=replace(done,status='expired_before_start',payload=None)
        result=ledger.settle(done)
        self.assertFalse(result.allow_capture_time_update)
        self.assertFalse(result.measurement_pending)
        self.assertEqual(result.measurements.cuda_stream_elapsed_s,0.)
        self.assertEqual(ledger.summary()['deep_jobs_admitted'],1)
        with self.assertRaises(AdmissionDenied):admit(ledger,item('d1'))

    def test_mode_alias_is_explicit_and_light_cost_is_not_optional_deep_cost(self):
        self.assertIs(canonical_schedule('move_lightweight'),ScheduleMode.MOVE_WITH_LIGHTWEIGHT)
        with self.assertRaises(AccountingError):canonical_schedule('invented_mode')
        ledger=JobAccounting(max_deep_budget_s=.1)
        packet=frame();work=WorkItem('l0',packet,20*S)
        ledger.admit(work,mode='light',schedule='move_lightweight',admitted_mono_ns=2*S)
        done=Completion('l0','light','f0',S,3*S,4*S,6*S,20*S,'ok',snapshot(),None,7*S)
        settled=ledger.settle(done,measurement(.5))
        self.assertTrue(settled.allow_capture_time_update)
        self.assertEqual(ledger.summary()['deep_budget_liability_s'],0.)
        self.assertEqual(ledger.summary()['known_cuda_stream_elapsed_s'],.5)
        with self.assertRaises(AccountingError):GpuMeasurement(cuda_stream_elapsed_s=.1)


if __name__=='__main__':unittest.main()
