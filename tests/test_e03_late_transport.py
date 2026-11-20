"""Synthetic parseable-image CPU contracts, never scientific performance."""
from dataclasses import replace
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from river_e03_perception import Candidate, FramePacket, LightSnapshot
from river_e03_late_transport import (CacheEntry, CaptureEvidenceStore,
    transport_candidate_probability)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


BASE_NS = 10_000_000_000


def fixture(step=0, shift=0, *, duplicate=False, absent=False, pose=None, masks=True):
    rng = np.random.default_rng(20261030)
    pixels = rng.integers(0,256,(128,128),dtype=np.uint8)
    image = np.zeros_like(pixels)
    if not absent:
        if shift >= 0:
            image[:,shift:] = pixels[:,:128-shift]
        else:
            image[:,:shift] = pixels[:,-shift:]
    rgb = np.repeat(image[:,:,None],3,axis=2)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf,format='JPEG',quality=100,subsampling=0)
    frame = FramePacket('sensor-'+str(step),step,step*.1,BASE_NS+step*100_000_000,
        buf.getvalue(),128,128,pose or (0.,)*6,(64.,0.,64.,0.,64.,64.,0.,0.,1.),
        'archived_replay_ingress')
    candidates = []
    if masks and not absent:
        mask = np.zeros((128,128),np.uint8)
        mask[32:88,32+shift:88+shift] = 1
        candidate = Candidate.from_observed(dict(xyxy=[32+shift,32,88+shift,88],
            confidence=.9,class_id=2,mask=mask),frame)
        candidates = [candidate,candidate] if duplicate else [candidate]
    return LightSnapshot(frame,tuple(candidates),step,(), '{}', (32.,32.,88.,88.) if step==0 else None)


def entry(snapshot, available=None):
    frame = snapshot.frame if isinstance(snapshot,LightSnapshot) else snapshot
    return CacheEntry(snapshot,'main/rgb',frame.jpeg_sha256,
                      available if available is not None else frame.capture_mono_ns+1_000_000)


class TestCausalInputAndReplacement(unittest.TestCase):
    def test_future_availability_rejected_before_any_flow(self):
        source,current = fixture(),fixture(1,4)
        decision = BASE_NS+110_000_000
        with patch.dict(sys.modules,{'cv2':None}):
            with self.assertRaises(ValueError):
                transport_candidate_probability(source,[.2,.8],
                    [entry(source),entry(current,decision+1)],delivered_mono_ns=decision)

    def test_missing_frame_and_current_candidate_cache_are_null(self):
        source,current = fixture(),fixture(2,8)
        result = transport_candidate_probability(source,[.2,.8],[entry(source),entry(current)],
                                                   delivered_mono_ns=BASE_NS+210_000_000)
        self.assertEqual(result.probability,(1.,0.))
        self.assertIn('missing',result.status)
        result = transport_candidate_probability(source,[.2,.8],
            [entry(source),entry(fixture(1,4).frame)],delivered_mono_ns=BASE_NS+110_000_000)
        self.assertEqual(result.probability,(1.,))

    def test_hash_camera_and_corrupt_mask_are_rejected(self):
        source = fixture()
        with self.assertRaises(ValueError):
            transport_candidate_probability(source,[.2,.8],
                [replace(entry(source),jpeg_sha256='bad')],delivered_mono_ns=BASE_NS+10_000_000)
        bad = replace(source,candidates=(replace(source.candidates[0],packed_mask=b'bad'),))
        with self.assertRaises(ValueError):
            transport_candidate_probability(bad,[.2,.8],[entry(bad)],delivered_mono_ns=BASE_NS+10_000_000)
        current = fixture(1,4)
        with self.assertRaises(ValueError):
            transport_candidate_probability(source,[.2,.8],
                [entry(source),replace(entry(current),camera_id='other/rgb')],
                delivered_mono_ns=BASE_NS+110_000_000)

    def test_deep_replaces_not_multiplies_and_repeated_delivery_is_idempotent(self):
        source = fixture()
        store = CaptureEvidenceStore()
        cheap = store.replace_capture_evidence(source,[.3,.7],camera_id='rgb',evidence_id='cheap',
                                               kind='cheap',available_mono_ns=BASE_NS+1)
        deep = store.replace_capture_evidence(source,[.1,.9],camera_id='rgb',evidence_id='deep',
                                              kind='deep',available_mono_ns=BASE_NS+5)
        repeated = store.replace_capture_evidence(source,[.1,.9],camera_id='rgb',evidence_id='deep',
                                                  kind='deep',available_mono_ns=BASE_NS+5)
        self.assertEqual(deep,repeated)
        self.assertEqual(deep.probability,(.1,.9))
        self.assertEqual(deep.revision,1)
        self.assertEqual(store.as_of(source,camera_id='rgb',delivered_mono_ns=BASE_NS+3),cheap)
        self.assertEqual(store.as_of(source,camera_id='rgb',delivered_mono_ns=BASE_NS+6),deep)
        with self.assertRaises(ValueError):
            store.replace_capture_evidence(source,[.2,.8],camera_id='rgb',evidence_id='deep',
                                            kind='deep',available_mono_ns=BASE_NS+5)
        with self.assertRaises(ValueError):
            store.replace_capture_evidence(source,[.1,.9],camera_id='rgb',evidence_id='renamed_duplicate',
                                            kind='deep',available_mono_ns=BASE_NS+6)


@unittest.skipUnless(HAS_CV2,'OpenCV is required for actual image-flow contracts')
class TestParsedImageTransport(unittest.TestCase):
    def apply(self, source, current, probability):
        return transport_candidate_probability(source,probability,[entry(source),entry(current)],
            delivered_mono_ns=current.frame.capture_mono_ns+10_000_000)

    def test_camera_motion_translation_carries_quality_without_changing_identity(self):
        source,current = fixture(),fixture(1,4,pose=(.2,0.,0.,0.,1.,0.))
        result = self.apply(source,current,[.2,.8])
        self.assertEqual(result.status,'transported')
        self.assertGreater(result.probability[1],.5)
        self.assertLessEqual(result.probability[1],.8+1e-10)
        self.assertAlmostEqual(sum(result.probability),1.)
        matrix = np.asarray(result.steps[-1]['capture_to_current_matrix'])
        np.testing.assert_allclose(np.array([.2,.8])@matrix,result.probability)

    def test_duplicate_current_masks_are_ambiguous_not_argmax(self):
        result = self.apply(fixture(),fixture(1,4,duplicate=True),[.2,.8])
        self.assertEqual(result.probability,(1.,0.,0.))

    def test_two_sources_cannot_copy_probability_to_one_candidate(self):
        result = self.apply(fixture(duplicate=True),fixture(1,4),[.1,.4,.5])
        self.assertEqual(result.probability,(1.,0.))

    def test_disappearance_loses_mass_to_null(self):
        result = self.apply(fixture(),fixture(1,absent=True),[.2,.8])
        self.assertEqual(result.probability,(1.,))

    def test_lost_mass_cannot_reappear_after_occlusion(self):
        source,hidden,current = fixture(),fixture(1,absent=True),fixture(2,8)
        result = transport_candidate_probability(source,[.2,.8],
            [entry(source),entry(hidden),entry(current)],delivered_mono_ns=BASE_NS+210_000_000)
        self.assertEqual(result.probability,(1.,0.))


if __name__=='__main__':
    unittest.main()
