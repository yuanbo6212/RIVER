"""Run RIVER on recorded CARLA observations or detect from the same JPEGs."""
import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from frozen_histogram_guard_v3 import make_guarded_identity
from rejection_readout_v1 import ARMS, RejectionReadout
from river_e02_control import CausalRepresentativeIdentity
from river_e03_perception import FramePacket, LightAdapter
from river_mobile_quality_v1_20261116 import MobileQualityDetector


class ArchivedProposals:
    def set_frame(self, folder, row):
        with np.load(folder / row['masks'], allow_pickle=False) as archive:
            packed = archive['packed_masks']
            height, width = row['height'], row['width']
            self.shape = (height, width)
            self.candidates = [dict(proposal, mask=np.unpackbits(mask)[:height * width].reshape(self.shape))
                for proposal, mask in zip(row['candidates'], packed, strict=True)]

    def predict(self, jpeg):
        return self.shape, self.candidates


class ImageDetector:
    def __init__(self, weights, device):
        from ultralytics import YOLO
        self.model = YOLO(str(weights.resolve(strict=True)))
        self.device = device

    def predict(self, jpeg):
        with Image.open(io.BytesIO(jpeg)) as image:
            rgb = image.convert('RGB')
        result = self.model.predict(source=rgb, imgsz=640, conf=.25, iou=.70,
            max_det=300, classes=[2], agnostic_nms=False, device=self.device,
            half=False, augment=False, retina_masks=True, verbose=False)[0]
        height, width = rgb.height, rgb.width
        if not len(result.boxes):
            return (height, width), []
        if result.masks is None or result.masks.data.shape != (len(result.boxes), height, width):
            raise ValueError('Detector must return one full-size mask per candidate')
        masks = (result.masks.data.cpu().numpy() > .5).astype(np.uint8)
        return (height, width), [dict(xyxy=box.xyxy[0].cpu().tolist(),
            confidence=float(box.conf.item()), class_id=int(box.cls.item()), mask=mask)
            for box, mask in zip(result.boxes, masks, strict=True)]


def replay(args):
    manifest = json.loads(args.sequence.read_text(encoding='utf-8'))
    rows = manifest['frames']
    steps = [row['source_step'] for row in rows]
    if not steps or steps[0] != 0 or steps != sorted(set(steps)):
        raise ValueError('Expected a strictly ordered sequence starting at step zero')
    if args.refresh and (args.weights is None or steps != list(range(230))):
        raise ValueError('Refresh comparisons require detector weights and all 230 observations')
    if args.weights is None:
        observer = detector = ArchivedProposals()
    else:
        observer = ImageDetector(args.weights, args.device)
        detector = MobileQualityDetector(observer, args.refresh) if args.refresh else observer
    guarded = make_guarded_identity(CausalRepresentativeIdentity)
    adapter = LightAdapter(detector, guarded(allow_development_use=True), manifest['initial_box'])
    readout = RejectionReadout(ROOT/'artifacts/e03_mode_readout_calibration_v1/models.json',
        allow_development_use=True, rejection_arm=args.rule)
    with args.output.open('x', encoding='utf-8') as output:
        for row in rows:
            folder = args.sequence.parent
            step = row['source_step']
            packet = FramePacket(str(row['frame_id']), step, row['capture_sim_s'],
                row['capture_mono_ns'], (folder/row['jpeg']).read_bytes(), row['width'],
                row['height'], tuple(row['pose']), tuple(row['intrinsic']), 'archived_replay_ingress')
            if row.get('jpeg_sha256') and packet.jpeg_sha256 != row['jpeg_sha256']:
                raise ValueError(f'JPEG hash mismatch at step {step}')
            if isinstance(observer, ArchivedProposals):
                observer.set_frame(folder, row)
            if args.refresh:
                detector.frame = packet
            snapshot = adapter(packet)
            if step == 0:
                readout.register_reference(snapshot)
            choice = readout.predict(snapshot, 'cheap')
            if isinstance(observer, ArchivedProposals):
                if row.get('candidate_sha256') and snapshot.candidate_sha256 != row['candidate_sha256']:
                    raise ValueError(f'Candidate hash mismatch at step {step}')
                expected = row.get('expected_probabilities')
                if args.rule == 'full_set' and expected is not None:
                    np.testing.assert_allclose(choice['probabilities'], expected, atol=1e-12, rtol=0)
            chosen = choice['selected_index']
            result = dict(step=step, selected_index=chosen,
                box=None if chosen is None else snapshot.candidates[chosen].xyxy,
                probabilities=choice['probabilities'], reason=choice['reason'])
            if args.refresh:
                detector.update_identity(choice)
                result.update(detector_calls=detector.calls, refresh=detector.audit[-1])
            output.write(json.dumps(result, allow_nan=False)+'\n')
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, default=ROOT/'examples/carla/sequence.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rule', choices=ARMS, default='full_set')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--refresh', choices=('full230','independent128','uniform128'))
    args = parser.parse_args()
    print(f'Replayed {replay(args)} observations to {args.output}')


if __name__ == '__main__':
    main()
