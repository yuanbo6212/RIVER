"""CPU interface tests only; fixtures are not inference/performance evidence."""
import io
import json
from pathlib import Path
from dataclasses import replace
import sys
import threading
import time
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from river_e03_perception import (Candidate, DeepAdapter, FramePacket, LatestFrameWorker,
                                   LightAdapter, LightSnapshot, WorkItem)


def frame(step):
    output = io.BytesIO()
    Image.fromarray(np.full((4, 6, 3), 80, np.uint8)).save(output, format="JPEG")
    return FramePacket("sensor-"+str(step), step, step*.1, time.monotonic_ns(),
                       output.getvalue(), 6, 4, (0.,)*6,
                       (3., 0., 3., 0., 3., 2., 0., 0., 1.), "archived_replay_ingress")


def work(packet, job=None):
    return WorkItem(job or packet.frame_id, packet, time.monotonic_ns()+5_000_000_000)


def result(worker):
    with worker._condition:
        ready = worker._condition.wait_for(lambda: worker._completed is not None, timeout=2.)
    if not ready:
        raise AssertionError("worker failed to produce bounded test result")
    return worker.poll()


class SequentialIdentity:
    """Only exercises the actual legacy consecutive-step calling convention."""
    def __init__(self):
        self.step = -1

    def update(self, rgb, candidates, step, pose, initial_box, fov):
        if step != self.step+1 or ((step == 0) != (initial_box is not None)):
            raise ValueError("legacy sequencing violation")
        self.step = step
        return None, dict(step=step, selected_index=None, gt_inputs=False)


class EmptyDetector:
    def predict(self, jpeg):
        return (4, 6), []


class TestE03Perception(unittest.TestCase):
    def test_source_skip_is_not_processed_ordinal(self):
        adapter = LightAdapter(EmptyDetector(), SequentialIdentity(), [1, 1, 4, 3])
        initial = adapter(frame(0))
        later = adapter(frame(4))
        self.assertEqual(initial.processed_ordinal, 0)
        self.assertEqual(later.processed_ordinal, 1)
        self.assertEqual(later.frame.source_step, 4)
        self.assertEqual(later.skipped_source_steps, (1, 2, 3))
        self.assertFalse(json.loads(later.identity_json)["complete_source_prefix"])

    def test_light_bootstrap_is_required(self):
        adapter = LightAdapter(EmptyDetector(), SequentialIdentity(), [1, 1, 4, 3])
        with self.assertRaises(ValueError):
            adapter(frame(1))

    def test_empty_deep_calls_no_feature_factory(self):
        adapter = LightAdapter(EmptyDetector(), SequentialIdentity(), [1, 1, 4, 3])
        initial = adapter(frame(0))
        def forbidden():
            raise AssertionError("DINO must not load/execute for zero candidates")
        deep = DeepAdapter(forbidden)
        deep.register_reference(initial)
        output = deep(initial)
        self.assertEqual(output["feature_extract_calls"], 0)
        self.assertEqual(output["candidate_sha256"], initial.candidate_sha256)
        self.assertEqual(output["jpeg_sha256"], initial.frame.jpeg_sha256)

    def test_snapshot_copies_original_mask(self):
        packet = frame(0)
        mask = np.ones((4, 6), np.uint8)
        candidate = Candidate.from_observed(dict(xyxy=[1, 1, 4, 3], confidence=.8,
                                                  class_id=2, mask=mask), packet)
        mask[:] = 0
        self.assertEqual(candidate.observed(packet)["mask"].sum(), 24)

    def test_bootstrap_cannot_be_replaced_owner_continues_latest_only(self):
        entered, release = threading.Event(), threading.Event()
        processed = []
        def blocking_contract(packet):
            processed.append(packet.source_step)
            if packet.source_step == 0:
                entered.set()
                if not release.wait(2.):
                    raise AssertionError("test release not delivered")
            return packet.frame_id  # interface echo, not fabricated neural evidence
        worker = LatestFrameWorker("light", blocking_contract, bootstrap=work(frame(0)))
        worker.start()
        try:
            self.assertTrue(entered.wait(2.))
            owner_ticks = []
            for step in range(1, 8):
                owner_ticks.append(step)
                worker.submit(work(frame(step)))
                self.assertLessEqual(worker.pending_count, 1)
            self.assertEqual(owner_ticks, list(range(1, 8)))
            release.set()
            first, latest = result(worker), result(worker)
            self.assertEqual(processed, [0, 7])
            self.assertEqual(first.frame_id, "sensor-0")
            self.assertEqual(latest.frame_id, "sensor-7")
            for item in (first, latest):
                stamps = [item.capture_mono_ns, item.enqueued_mono_ns,
                          item.compute_started_mono_ns, item.compute_ended_mono_ns,
                          item.delivered_mono_ns]
                self.assertEqual(stamps, sorted(stamps))
            audit = worker.drain_audit()
            self.assertEqual(sum(v["kind"] == "replaced_before_start" for v in audit), 6)
        finally:
            release.set()
            self.assertTrue(worker.close())

    def test_expired_pending_is_retained_without_processor_call(self):
        packet = frame(0)
        worker = LatestFrameWorker("light", lambda _: self.fail("expired job executed"))
        self.assertFalse(worker.submit(WorkItem("expired", packet, time.monotonic_ns()-1)))
        self.assertEqual(worker.pending_count, 0)
        self.assertEqual(worker.drain_audit()[0]["kind"], "expired_before_enqueue")
        self.assertTrue(worker.close())

    def test_error_is_not_replaced_by_success(self):
        def broken(_):
            raise RuntimeError("explicit technical failure")
        worker = LatestFrameWorker("light", broken)
        worker.start()
        try:
            worker.submit(work(frame(0)))
            failure = result(worker)
            self.assertEqual(failure.status, "error")
            self.assertIn("explicit technical failure", failure.error)
            with self.assertRaises(ValueError):
                worker.submit(work(frame(1)))
        finally:
            self.assertTrue(worker.close())

    def test_bootstrap_and_workitem_reject_invalid_provenance(self):
        packet = frame(0)
        with self.assertRaises(ValueError):
            WorkItem(7, packet, time.monotonic_ns()+1_000_000)
        with self.assertRaises(ValueError):
            WorkItem("bad", object(), time.monotonic_ns()+1_000_000)
        with self.assertRaises(ValueError):
            WorkItem("bad", packet, 1.5)
        future = replace(packet, capture_mono_ns=time.monotonic_ns()+10_000_000_000)
        with self.assertRaises(ValueError):
            LatestFrameWorker("light", lambda x: x, bootstrap=work(future))

    def test_reject_K_that_fov_interface_cannot_represent(self):
        packet = frame(0)
        with self.assertRaises(ValueError):
            replace(packet, intrinsic=(3., 1., 3., 0., 3., 2., 0., 0., 1.))
        with self.assertRaises(ValueError):
            replace(packet, intrinsic=(3., 0., 3., 0., 4., 2., 0., 0., 1.))


if __name__ == "__main__":
    unittest.main()
