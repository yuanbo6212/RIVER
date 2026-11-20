"""Synthetic contract checks only: no simulator or measured runtime claims."""

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from river_runtime import (ActionRequest, BudgetExceeded, CapturedFrame, CausalRuntime,
                           ConstantVelocityReplay, ContractError, InferenceResult,
                           KinematicState, MotionConstraintViolation, PositionEvidence,
                           ResultRejected, ScheduleMode)


def runtime(**overrides):
    options = dict(task_deadline=10.0, max_jobs=10, compute_budget_seconds=20.0)
    options.update(overrides)
    return CausalRuntime(ConstantVelocityReplay(), KinematicState((0.0, 0.0), (2.0, 0.0)),
                         **options)


def result(job, *, result_id=None, evidence_id=None, end=1.0, delivery=1.0,
           position=(10.0, 0.0)):
    return InferenceResult(result_id or f"{job.job_id}/result",
                           evidence_id or f"{job.job_id}/evidence", job.job_id,
                           job.frame_id, job.captured_at, job.compute_started_at,
                           end, delivery, PositionEvidence(position))


def compute_job(rt, job_id="j", frame_id="f", *, deadline=5.0, compute_bound=3.0,
                action=None):
    return rt.submit_job(job_id, frame_id, mode=ScheduleMode.COMPUTE_THEN_MOVE,
                         compute_bound=compute_bound, deadline=deadline,
                         action_policy=action or (lambda state: ActionRequest((0.0, 0.0))))


class CausalObservationTests(unittest.TestCase):
    def test_future_frames_and_clock_reversal_are_rejected(self):
        rt = runtime()
        with self.assertRaises(ContractError):
            rt.register_frame(CapturedFrame("future", 0.01))
        rt.advance_to(2.0)
        with self.assertRaises(ContractError):
            rt.advance_to(1.0)
        self.assertEqual(rt.now, 2.0)

    def test_capture_compute_delivery_action_effect_are_distinct(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        rt.advance_to(0.5)
        job = compute_job(rt, action=lambda state: ActionRequest((1.0, 0.0), 0.5))
        rt.schedule_result(result(job, end=1.5, delivery=2.0))
        rt.advance_to(1.9)
        self.assertEqual(rt.actions, {})
        rt.advance_to(2.0)
        action = rt.actions["j/after_compute"]
        self.assertEqual((job.captured_at, job.compute_started_at), (0.0, 0.5))
        record = rt.jobs["j"]
        self.assertEqual((record.compute_ended_at, record.delivered_at,
                          record.belief_updated_at, record.used_for_action_at),
                         (1.5, 2.0, 2.0, 2.0))
        self.assertEqual((action.issued_at, action.effect_due_at, action.effect_at),
                         (2.0, 2.5, None))
        rt.schedule_motion(action.command_id, (0.75, 0.0), effect_at=2.6)
        rt.advance_to(2.6)
        self.assertEqual(rt.actions[action.command_id].effect_at, 2.6)
        self.assertEqual(rt.actions[action.command_id].realized_displacement, (0.75, 0.0))

    def test_stale_bbox_counterexample_is_propagated_not_relabelled_current(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        job = compute_job(rt)
        rt.schedule_result(result(job, end=1.0, delivery=2.0, position=(10.0, 0.0)))
        rt.advance_to(2.0)
        # Captured position 10 at t=0 with v=2 means current position 14 at t=2.
        # Relabelling the stale box as current would silently return 10.
        self.assertEqual(rt.state.position, (14.0, 0.0))
        self.assertNotEqual(rt.state.position, (10.0, 0.0))
        self.assertEqual(rt.state.timestamp, 2.0)

    def test_delayed_result_replays_intervening_measurement_and_camera_motion(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        job = compute_job(rt)
        rt.issue_action("camera", ActionRequest((1.0, 0.0)), deadline=3.0)
        rt.schedule_motion("camera", (1.0, 0.0), effect_at=1.5)
        rt.advance_to(1.0)
        rt.register_frame(CapturedFrame("new", 1.0))
        rt.accept_lightweight("new", "new/evidence", PositionEvidence((100.0, 0.0)))
        rt.schedule_result(result(job, end=0.5, delivery=2.0))
        rt.advance_to(2.0)
        # Rewind: obs(0)=10 -> propagate to obs(1)=100 -> camera(1.5)=-1 -> t2.
        # Merely propagating the old result forward would instead give 13.
        self.assertEqual(rt.state.position, (101.0, 0.0))
        self.assertEqual(rt.replay_trace, ("j/evidence", "new/evidence"))
        self.assertEqual(len(rt.actions), 2)  # Replaying did not reissue camera.

    def test_out_of_order_delivery_matches_capture_order(self):
        def run(deliver_old_first):
            rt = runtime()
            rt.register_frame(CapturedFrame("old", 0.0))
            old = compute_job(rt, "old-job", "old", compute_bound=5.0)
            rt.advance_to(1.0)
            rt.register_frame(CapturedFrame("new", 1.0))
            new = compute_job(rt, "new-job", "new")
            old_delivery = 1.5 if deliver_old_first else 3.0
            new_delivery = 3.0 if deliver_old_first else 1.5
            rt.schedule_result(result(old, end=1.2, delivery=old_delivery))
            rt.schedule_result(result(new, end=1.2, delivery=new_delivery, position=(50.0, 0.0)))
            rt.advance_to(4.0)
            return rt
        first, reversed_order = run(True), run(False)
        self.assertEqual(first.state, reversed_order.state)
        self.assertEqual(first.state.position, (56.0, 0.0))
        self.assertEqual(first.replay_trace, reversed_order.replay_trace)

    def test_future_result_is_not_usable_before_delivery(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        job = compute_job(rt)
        pending = result(job, end=1.0, delivery=2.0)
        rt.schedule_result(pending)
        with self.assertRaises(ResultRejected):
            rt.deliver_result(pending)
        rt.advance_to(1.0)
        self.assertEqual(rt.state.position, (2.0, 0.0))
        self.assertEqual(rt.accepted_evidence_ids, frozenset())
        self.assertEqual(rt.ledger.reported_compute_seconds, 0.0)

    def test_result_provenance_and_impossible_clock_order_rejected(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        job = compute_job(rt)
        with self.assertRaises(ContractError):
            result(job, end=2.0, delivery=1.0)
        rt.advance_to(1.0)
        with self.assertRaises(ResultRejected):
            rt.deliver_result(replace(result(job), frame_id="wrong"))
        with self.assertRaises(ResultRejected):
            rt.deliver_result(replace(result(job), compute_started_at=0.1))
        self.assertEqual(rt.accepted_evidence_ids, frozenset())

    def test_result_and_evidence_cannot_be_reused(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        job = compute_job(rt)
        rt.advance_to(1.0)
        completed = result(job)
        rt.deliver_result(completed)
        before = (rt.state, rt.ledger, rt.actions)
        with self.assertRaises(ResultRejected):
            rt.deliver_result(completed)
        with self.assertRaises(ResultRejected):
            rt.deliver_result(replace(completed, result_id="new-id"))
        with self.assertRaises(ResultRejected):
            rt.accept_lightweight("f", completed.evidence_id, PositionEvidence((99.0, 0.0)))
        self.assertEqual((rt.state, rt.ledger, rt.actions), before)

    def test_exactly_once_logical_evidence_with_non_idempotent_estimator(self):
        @dataclass(frozen=True)
        class SumState:
            total: int
            timestamp: float

        class SumReplay:
            def propagate(self, state, from_at, to_at):
                self.assert_stamp(state, from_at)
                return SumState(state.total, to_at)

            def assert_stamp(self, state, stamp):
                if state.timestamp != stamp:
                    raise ContractError("bad timestamp")

            def observe(self, state, evidence, captured_at):
                self.assert_stamp(state, captured_at)
                return SumState(state.total + evidence, captured_at)

            def camera_motion(self, state, displacement, effect_at):
                return state

        rt = CausalRuntime(SumReplay(), SumState(0, 0.0), task_deadline=5.0,
                           max_jobs=2, compute_budget_seconds=3.0)
        rt.register_frame(CapturedFrame("old", 0.0))
        old = compute_job(rt, frame_id="old", deadline=4.0)
        rt.accept_lightweight("old", "initial", 3)
        rt.advance_to(1.0)
        rt.register_frame(CapturedFrame("new", 1.0))
        rt.accept_lightweight("new", "later", 5)
        rt.schedule_result(replace(result(old, end=1.0, delivery=2.0), evidence=2))
        rt.advance_to(2.0)
        rt.advance_to(3.0)
        self.assertEqual(rt.state.total, 10)
        self.assertEqual(len(rt.replay_trace), 3)
        self.assertEqual(len(set(rt.replay_trace)), 3)
        self.assertEqual(len(rt.actions), 1)
        self.assertEqual(rt.ledger.jobs_submitted, 1)


class SchedulingTests(unittest.TestCase):
    def test_compute_in_flight_overlaps_action_and_does_not_reissue_it(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        rt.accept_lightweight("f", "hsv", PositionEvidence((0.0, 0.0)))
        job = rt.submit_job("j", "f", mode=ScheduleMode.COMPUTE_IN_FLIGHT,
                            compute_bound=3.0, deadline=4.0,
                            in_flight_action=ActionRequest((1.0, 0.0), 0.5),
                            evidence_ids=("hsv",))
        rt.schedule_motion("j/in_flight", (0.5, 0.0), effect_at=0.5)
        rt.schedule_result(result(job, end=2.0, delivery=3.0))
        rt.advance_to(3.0)
        self.assertEqual(len(rt.actions), 1)
        self.assertEqual(rt.actions["j/in_flight"].issued_at, 0.0)
        self.assertEqual(rt.actions["j/in_flight"].effect_at, 0.5)
        self.assertIsNone(rt.jobs["j"].used_for_action_at)
        self.assertEqual(rt.jobs["j"].belief_updated_at, 3.0)
        self.assertEqual(rt.ledger.elapsed_task_seconds, 3.0)
        self.assertEqual(rt.ledger.reported_compute_seconds, 2.0)
        self.assertEqual(rt.state.position, (15.5, 0.0))

    def test_lightweight_moves_immediately_without_extra_job(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        command = rt.move_with_lightweight("move", "f", "hsv", PositionEvidence((1.0, 0.0)),
                                           ActionRequest((0.25, 0.0)), deadline=1.0)
        self.assertEqual(command.issued_at, 0.0)
        self.assertEqual(command.evidence_ids, ("hsv",))
        self.assertEqual(rt.ledger.jobs_submitted, 0)
        self.assertEqual(rt.ledger.committed_compute_seconds, 0.0)

    def test_in_flight_result_first_action_use_is_logged_separately(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        job = rt.submit_job("j", "f", mode=ScheduleMode.COMPUTE_IN_FLIGHT,
                            compute_bound=1.0, deadline=3.0,
                            in_flight_action=ActionRequest((0.0, 0.0)))
        rt.schedule_result(result(job, end=1.0, delivery=1.0))
        rt.advance_to(2.0)
        self.assertEqual(rt.jobs["j"].belief_updated_at, 1.0)
        self.assertIsNone(rt.jobs["j"].used_for_action_at)
        rt.issue_action("later", ActionRequest((0.0, 0.0)), deadline=3.0,
                        evidence_ids=("j/evidence",))
        self.assertEqual(rt.jobs["j"].used_for_action_at, 2.0)
        rt.advance_to(2.5)
        rt.issue_action("later-again", ActionRequest((0.0, 0.0)), deadline=3.0,
                        evidence_ids=("j/evidence",))
        self.assertEqual(rt.jobs["j"].used_for_action_at, 2.0)

    def test_timeout_fixed_deadline_fallback_and_late_result_discard(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        job = compute_job(rt, deadline=1.0, compute_bound=3.0)
        rt.schedule_result(result(job, end=2.0, delivery=3.0))
        rt.advance_to(1.0)
        self.assertEqual(rt.jobs["j"].status, "timed_out")
        fallback = rt.actions["j/fallback"]
        self.assertTrue(fallback.is_fallback)
        self.assertEqual(fallback.issued_at, 1.0)
        self.assertEqual(fallback.command, (0.0, 0.0))
        rt.advance_to(3.0)
        self.assertEqual(rt.accepted_evidence_ids, frozenset())
        self.assertEqual(len(rt.actions), 1)
        self.assertEqual(rt.ledger.deadline_misses, 1)
        self.assertEqual(rt.ledger.committed_compute_seconds, 3.0)
        self.assertEqual(rt.ledger.reported_compute_seconds, 2.0)
        self.assertTrue(any(event.kind == "result_rejected" for event in rt.audit))

    def test_response_delay_that_cannot_meet_deadline_falls_back(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        job = compute_job(rt, deadline=1.0, action=lambda state: ActionRequest((1.0, 0.0), 0.6))
        rt.schedule_result(result(job, end=0.5, delivery=0.5))
        rt.advance_to(2.0)
        self.assertEqual(set(rt.actions), {"j/fallback"})
        self.assertEqual(rt.actions["j/fallback"].issued_at, 0.5)
        self.assertEqual(rt.ledger.deadline_misses, 1)
        self.assertEqual(rt.jobs["j"].deadline, 1.0)

    def test_result_at_deadline_is_usable_with_zero_response_delay(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        job = compute_job(rt, deadline=1.0)
        rt.schedule_result(result(job, end=1.0, delivery=1.0))
        rt.advance_to(1.0)
        self.assertEqual(rt.jobs["j"].status, "completed")
        self.assertEqual(set(rt.actions), {"j/after_compute"})
        self.assertEqual(rt.ledger.deadline_misses, 0)

    def test_same_time_motion_is_available_to_result_policy(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        seen = []

        def policy(state):
            seen.append(state.position)
            return ActionRequest((0.0, 0.0))

        job = compute_job(rt, action=policy)
        rt.issue_action("camera", ActionRequest((1.0, 0.0)), deadline=2.0)
        rt.schedule_result(result(job, end=1.0, delivery=1.0))
        rt.schedule_motion("camera", (1.0, 0.0), effect_at=1.0)
        rt.advance_to(1.0)
        self.assertEqual(seen, [(11.0, 0.0)])

    def test_task_horizon_never_extends_to_finish_slow_work(self):
        rt = runtime(task_deadline=2.0)
        rt.register_frame(CapturedFrame("f", 0.0))
        job = compute_job(rt, deadline=2.0)
        with self.assertRaises(ContractError):
            rt.schedule_result(result(job, end=2.5, delivery=3.0))
        with self.assertRaises(ContractError):
            rt.advance_to(3.0)
        rt.advance_to(2.0)
        self.assertEqual(rt.jobs["j"].status, "timed_out")
        self.assertEqual(rt.ledger.elapsed_task_seconds, 2.0)

    def test_scheduling_mode_arguments_are_explicit(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        with self.assertRaises(ContractError):
            rt.submit_job("j", "f", mode=ScheduleMode.COMPUTE_IN_FLIGHT,
                          compute_bound=1.0, deadline=2.0)
        with self.assertRaises(ContractError):
            rt.submit_job("j", "f", mode=ScheduleMode.MOVE_WITH_LIGHTWEIGHT,
                          compute_bound=1.0, deadline=2.0)
        self.assertEqual(rt.ledger.jobs_submitted, 0)


class ResourceTests(unittest.TestCase):
    def test_job_count_and_compute_bound_are_admission_budgets(self):
        rt = runtime(max_jobs=2, compute_budget_seconds=2.0)
        rt.register_frame(CapturedFrame("f", 0.0))
        compute_job(rt, "first", compute_bound=1.5)
        with self.assertRaises(BudgetExceeded):
            compute_job(rt, "too-expensive", compute_bound=0.6)
        self.assertEqual(rt.ledger.jobs_submitted, 1)
        compute_job(rt, "second", compute_bound=0.5)
        with self.assertRaises(BudgetExceeded):
            compute_job(rt, "too-many", compute_bound=0.0)
        self.assertEqual(rt.ledger.jobs_submitted, 2)
        self.assertEqual(rt.ledger.committed_compute_seconds, 2.0)

    def test_timeout_does_not_refund_a_still_running_job(self):
        rt = runtime(compute_budget_seconds=1.0)
        rt.register_frame(CapturedFrame("f", 0.0))
        compute_job(rt, deadline=0.5, compute_bound=1.0)
        rt.advance_to(0.5)
        with self.assertRaises(BudgetExceeded):
            compute_job(rt, "new", deadline=1.5, compute_bound=0.1)
        self.assertEqual(rt.ledger.committed_compute_seconds, 1.0)

    def test_compute_bound_overrun_is_rejected_explicitly(self):
        rt = runtime()
        rt.register_frame(CapturedFrame("f", 0.0))
        job = compute_job(rt, compute_bound=0.5)
        rt.advance_to(1.0)
        with self.assertRaisesRegex(ResultRejected, "bound exceeded"):
            rt.deliver_result(result(job))
        self.assertEqual(rt.accepted_evidence_ids, frozenset())

    def test_command_reservations_and_realized_path_are_separate(self):
        rt = runtime(total_motion_cap=3.0)
        rt.issue_action("first", ActionRequest((2.0, 0.0)), deadline=4.0)
        with self.assertRaises(BudgetExceeded):
            rt.issue_action("too-soon", ActionRequest((1.1, 0.0)), deadline=4.0)
        rt.advance_to(1.0)
        rt.record_motion("first", (1.0, 0.0), effect_at=1.0)
        rt.issue_action("second", ActionRequest((-2.0, 0.0)), deadline=4.0)
        self.assertEqual(rt.ledger.commanded_distance, 4.0)
        self.assertEqual(rt.ledger.realized_distance, 1.0)
        self.assertEqual(rt.ledger.reserved_motion_distance, 2.0)
        with self.assertRaises(BudgetExceeded):
            rt.issue_action("too-far", ActionRequest((0.01, 0.0)), deadline=4.0)
        rt.advance_to(2.0)
        rt.record_motion("second", (-1.5, 0.0), effect_at=2.0)
        self.assertEqual(rt.ledger.realized_distance, 2.5)
        self.assertEqual(rt.ledger.reserved_motion_distance, 0.0)

    def test_path_length_does_not_cancel_opposite_motions(self):
        rt = runtime(total_motion_cap=2.0)
        rt.issue_action("out", ActionRequest((1.0, 0.0)), deadline=3.0)
        rt.issue_action("back", ActionRequest((-1.0, 0.0)), deadline=3.0)
        rt.advance_to(1.0)
        rt.record_motion("out", (1.0, 0.0), effect_at=1.0)
        rt.advance_to(2.0)
        rt.record_motion("back", (-1.0, 0.0), effect_at=2.0)
        self.assertEqual(rt.ledger.realized_distance, 2.0)
        with self.assertRaises(BudgetExceeded):
            rt.issue_action("extra", ActionRequest((0.01, 0.0)), deadline=3.0)

    def test_actual_motion_violation_is_retained_and_future_motion_inhibited(self):
        rt = runtime(total_motion_cap=1.0)
        rt.issue_action("move", ActionRequest((1.0, 0.0)), deadline=3.0)
        rt.advance_to(1.0)
        with self.assertRaises(MotionConstraintViolation):
            rt.record_motion("move", (1.2, 0.0), effect_at=1.0)
        self.assertAlmostEqual(rt.ledger.realized_distance, 1.2)
        self.assertAlmostEqual(rt.state.position[0], 0.8)
        self.assertTrue(rt.motion_fault)
        with self.assertRaises(BudgetExceeded):
            rt.issue_action("extra", ActionRequest((0.01, 0.0)), deadline=3.0)
        rt.issue_action("hold", ActionRequest((0.0, 0.0)), deadline=3.0)

    def test_realized_late_effect_is_reported_instead_of_hidden(self):
        rt = runtime()
        rt.issue_action("move", ActionRequest((1.0, 0.0), 0.5), deadline=1.0)
        rt.advance_to(2.0)
        rt.record_motion("move", (1.0, 0.0), effect_at=2.0)
        self.assertEqual(rt.ledger.deadline_misses, 1)
        self.assertEqual(rt.actions["move"].effect_due_at, 0.5)
        self.assertEqual(rt.actions["move"].effect_at, 2.0)

    def test_motion_effect_cannot_be_reused_or_scheduled_twice(self):
        rt = runtime()
        rt.issue_action("move", ActionRequest((1.0, 0.0)), deadline=2.0)
        rt.schedule_motion("move", (0.5, 0.0), effect_at=1.0)
        with self.assertRaises(ContractError):
            rt.schedule_motion("move", (0.5, 0.0), effect_at=1.0)
        rt.advance_to(1.0)
        with self.assertRaises(ContractError):
            rt.record_motion("move", (0.5, 0.0), effect_at=1.0)
        self.assertEqual(rt.ledger.realized_distance, 0.5)

    def test_invalid_numeric_inputs_and_initial_timestamp_rejected(self):
        for value in (float("nan"), float("inf"), -1.0):
            with self.assertRaises(ContractError):
                CapturedFrame("f", value)
            with self.assertRaises(ContractError):
                runtime(total_motion_cap=value)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ContractError):
                ActionRequest((value, 0.0))
        with self.assertRaises(ContractError):
            CausalRuntime(ConstantVelocityReplay(), KinematicState((0.0,), (1.0,), 1.0),
                          task_deadline=2.0, max_jobs=1, compute_budget_seconds=1.0)


if __name__ == "__main__":
    unittest.main()
