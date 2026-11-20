"""CPU, causal image association for delayed candidate evidence; no GT/model/3D.

This is a conservative LK transport, not a physical identity posterior or a
complete replay estimator. Missing support is absorbing null probability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np

from river_e03_perception import Candidate, FramePacket, LightSnapshot


@dataclass(frozen=True)
class TransportConfig:
    max_candidates: int = 16
    max_steps: int = 20
    max_age_s: float = 2.
    max_current_age_s: float = .15
    frame_interval_s: float = .1
    interval_tolerance_s: float = .002
    max_points: int = 24
    min_points: int = 6
    quality: float = .02
    min_distance: float = 3.
    fb_error_px: float = 1.
    patch_error: float = 20.
    min_survival: float = .6
    match_support: float = .75
    second_support: float = .10


@dataclass(frozen=True)
class CacheEntry:
    payload: FramePacket | LightSnapshot
    camera_id: str
    jpeg_sha256: str
    available_mono_ns: int

    @property
    def frame(self):
        return self.payload.frame if isinstance(self.payload, LightSnapshot) else self.payload


@dataclass(frozen=True)
class TransportResult:
    probability: tuple[float, ...]
    current_frame_id: str
    capture_to_delivery_age_s: float
    current_age_s: float
    status: str
    steps: tuple[dict, ...]
    parameters: dict


def _probability(value, count):
    probability = np.array(value, dtype=float, copy=True)
    if (probability.shape != (count+1,) or not np.isfinite(probability).all() or
            np.any(probability < 0) or np.any(probability > 1) or
            not math.isclose(float(probability.sum()), 1., abs_tol=1e-10, rel_tol=0)):
        raise ValueError('finite normalized null-plus-candidate distribution required')
    return tuple(float(x) for x in probability)


def _validate_snapshot(snapshot):
    if not isinstance(snapshot, LightSnapshot) or not isinstance(snapshot.frame, FramePacket):
        raise ValueError('original complete LightSnapshot required')
    frame = snapshot.frame
    length = (frame.width*frame.height+7)//8
    for candidate in snapshot.candidates:
        if (not isinstance(candidate, Candidate) or not isinstance(candidate.packed_mask, bytes) or
                len(candidate.packed_mask) != length or not isinstance(candidate.xyxy, tuple)):
            raise ValueError('candidate mask/rectangle ownership or size invalid')
        if Candidate.from_observed(candidate.observed(frame), frame) != candidate:
            raise ValueError('candidate content/padding not canonical')


def _frame_binding(frame):
    return (frame.frame_id, frame.source_step, frame.capture_sim_s, frame.capture_mono_ns,
            frame.width, frame.height, frame.pose, frame.intrinsic, frame.capture_origin,
            frame.jpeg_sha256)


def _validate_cache(capture, cache, delivery):
    if type(delivery) is not int or delivery < capture.frame.capture_mono_ns or not cache:
        raise ValueError('actual delivery and nonempty causal cache required')
    if not isinstance(cache[0], CacheEntry):
        raise ValueError('explicit cache availability receipt required')
    camera = cache[0].camera_id
    seen = set()
    for entry in cache:
        if not isinstance(entry, CacheEntry) or not isinstance(entry.frame, FramePacket):
            raise ValueError('explicit cache availability receipt required')
        frame = entry.frame
        if (not isinstance(camera, str) or not camera or entry.camera_id != camera or
                entry.jpeg_sha256 != frame.jpeg_sha256):
            raise ValueError('camera/JPEG source binding mismatch')
        if (type(entry.available_mono_ns) is not int or
                not frame.capture_mono_ns <= entry.available_mono_ns <= delivery):
            raise ValueError('future or invalid image/candidate availability')
        if frame.frame_id in seen:
            raise ValueError('duplicate or conflicting cached frame ID')
        seen.add(frame.frame_id)
        if (frame.width, frame.height) != (capture.frame.width, capture.frame.height) or not np.allclose(
                frame.intrinsic, capture.frame.intrinsic, rtol=0, atol=1e-9):
            raise ValueError('camera shape/intrinsic mismatch')
        if isinstance(entry.payload, LightSnapshot):
            _validate_snapshot(entry.payload)
    if _frame_binding(cache[0].frame) != _frame_binding(capture.frame):
        raise ValueError('cache origin differs from capture frame/hash/pose/time')
    if isinstance(cache[0].payload, LightSnapshot) and cache[0].payload.candidate_sha256 != capture.candidate_sha256:
        raise ValueError('cache origin candidates differ from capture evidence')


def transport_candidate_probability(capture: LightSnapshot, probability,
        cache: Sequence[CacheEntry], *, delivered_mono_ns: int,
        config: TransportConfig = TransportConfig()) -> TransportResult:
    """Transport only past pixels and already delivered candidate masks.

    Intermediate raw images permit LK only. Their absence of a candidate
    snapshot is not a detector miss or independent visibility certificate.
    """
    _validate_snapshot(capture)
    original = _probability(probability, len(capture.candidates))
    cache = tuple(cache)
    _validate_cache(capture, cache, delivered_mono_ns)  # before any LK/decode
    current = cache[-1].payload
    count = len(current.candidates) if isinstance(current, LightSnapshot) else 0
    age = (delivered_mono_ns-capture.frame.capture_mono_ns)/1e9
    current_age = (delivered_mono_ns-cache[-1].frame.capture_mono_ns)/1e9
    audit = []

    def finish(weights=None, indices=None, status='transported'):
        output = np.zeros(count+1, float)
        if weights is not None:
            for weight, index in zip(weights, indices):
                if index is not None:
                    output[index+1] += weight
        output[0] = 1.-float(output[1:].sum())
        if output[0] < -1e-10:
            raise ValueError('probability conservation violated')
        output[0] = max(0., output[0])
        return TransportResult(tuple(float(x) for x in output), cache[-1].frame.frame_id,
                               age, current_age, status, tuple(audit), asdict(config))

    def unsupported(reason):
        audit.append(dict(reason=reason, null_probability=1., candidate_probability=0.))
        return finish(status='unsupported_'+reason)

    if len(capture.candidates) > config.max_candidates:
        return unsupported('source_candidate_budget')
    if len(cache)-1 > config.max_steps or age > config.max_age_s:
        return unsupported('transport_horizon')
    if current_age > config.max_current_age_s:
        return unsupported('current_image_stale')
    if not isinstance(current, LightSnapshot):
        return unsupported('current_candidates_unavailable')
    for previous, following in zip(cache, cache[1:]):
        a, b = previous.frame, following.frame
        if (b.source_step != a.source_step+1 or b.capture_mono_ns <= a.capture_mono_ns or
                abs((b.capture_sim_s-a.capture_sim_s)-config.frame_interval_s)>config.interval_tolerance_s):
            return unsupported('missing_or_nonmonotone_frame')
    if len(cache) == 1:
        return TransportResult(original, current.frame.frame_id, age, current_age,
                               'same_capture_frame', (), asdict(config))

    import cv2
    def gray(frame):
        return cv2.cvtColor(frame.rgb(), cv2.COLOR_RGB2GRAY)
    previous_gray = gray(capture.frame)
    weights = np.array(original[1:], float)
    indices = list(range(len(capture.candidates)))
    tracks = []
    for i, candidate in enumerate(capture.candidates):
        mask = candidate.observed(capture.frame)['mask']
        mask = cv2.erode(mask, np.ones((3,3), np.uint8))*255
        points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=config.max_points,
            qualityLevel=config.quality, minDistance=config.min_distance, mask=mask, blockSize=3)
        if points is None or len(points) < config.min_points:
            points = np.empty((0,1,2), np.float32)
            weights[i] = 0.
        tracks.append(points)
    audit.append(dict(kind='initial_mask_features', frame_id=capture.frame.frame_id,
        counts=[len(x) for x in tracks], candidate_mass=float(weights.sum()),
        null_probability=1.-float(weights.sum()), no_point_replenishment=True))
    lk = dict(winSize=(21,21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    for entry in cache[1:]:
        frame = entry.frame
        image = gray(frame)
        flow_rows = []
        for i, points in enumerate(tracks):
            before_count, before_mass = len(points), float(weights[i])
            if not before_count or not before_mass:
                tracks[i] = np.empty((0,1,2), np.float32)
                flow_rows.append(dict(source=i, before_points=before_count, valid_points=0,
                                      before_mass=before_mass, retained_mass=0.))
                continue
            try:
                forward, ok, error = cv2.calcOpticalFlowPyrLK(previous_gray, image, points, None, **lk)
            except cv2.error:
                forward = ok = error = None
            if forward is None or ok is None or error is None:
                valid = np.zeros(before_count, bool)
                forward = points.copy()
                fb = np.full(before_count, np.inf)
            else:
                try:
                    backward, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(image, previous_gray, forward, None, **lk)
                except cv2.error:
                    backward = reverse_ok = None
                fb = np.linalg.norm((backward-points).reshape(-1,2), axis=1) if backward is not None else np.full(before_count,np.inf)
                positions = forward.reshape(-1,2)
                valid = (ok.ravel().astype(bool) & np.isfinite(positions).all(1) &
                    np.isfinite(error.ravel()) & (error.ravel()<=config.patch_error) &
                    np.isfinite(fb) & (fb<=config.fb_error_px) &
                    (positions[:,0]>=0) & (positions[:,0]<frame.width) &
                    (positions[:,1]>=0) & (positions[:,1]<frame.height))
                valid &= reverse_ok.ravel().astype(bool) if reverse_ok is not None else False
            valid_count = int(valid.sum())
            survival = valid_count/before_count
            if valid_count < config.min_points or survival < config.min_survival:
                valid[:] = False
                weights[i] = 0.
            else:
                weights[i] *= survival
            tracks[i] = forward[valid].astype(np.float32)
            finite_fb = fb[np.isfinite(fb)]
            flow_rows.append(dict(source=i, before_points=before_count, valid_points=valid_count,
                carried_points=len(tracks[i]), survival_fraction=survival, before_mass=before_mass,
                retained_mass=float(weights[i]), maximum_finite_fb_error_px=float(finite_fb.max()) if len(finite_fb) else None))
        mapping = []
        if isinstance(entry.payload, LightSnapshot):
            snapshot = entry.payload
            if len(snapshot.candidates) > config.max_candidates:
                return unsupported('current_candidate_budget')
            masks = [candidate.observed(frame)['mask'] for candidate in snapshot.candidates]
            proposals, memberships = {}, {}
            for i, points in enumerate(tracks):
                support = np.zeros(len(masks), float)
                exclusive = np.empty((0,len(masks)), bool)
                if len(points) and masks and weights[i]:
                    xy = np.rint(points.reshape(-1,2)).astype(int)
                    xy[:,0] = np.clip(xy[:,0],0,frame.width-1)
                    xy[:,1] = np.clip(xy[:,1],0,frame.height-1)
                    hits = np.column_stack([mask[xy[:,1],xy[:,0]].astype(bool) for mask in masks])
                    exclusive = hits & (hits.sum(1)==1)[:,None]
                    support = exclusive.sum(0)/len(points)
                    ranking = np.sort(support)
                    if ranking[-1]>=config.match_support and (len(ranking)==1 or ranking[-2]<=config.second_support):
                        proposals[i] = int(np.argmax(support))
                memberships[i] = (support, exclusive)
            target_counts = {target:sum(value==target for value in proposals.values()) for target in proposals.values()}
            for i, points in enumerate(tracks):
                support, exclusive = memberships[i]
                target = proposals.get(i)
                before_mass = float(weights[i])
                if target is None or target_counts[target] != 1:
                    weights[i] = 0.
                    tracks[i] = np.empty((0,1,2), np.float32)
                    indices[i] = None
                else:
                    weights[i] *= float(support[target])
                    tracks[i] = points[exclusive[:,target]]
                    indices[i] = target
                mapping.append(dict(capture_candidate=i, current_candidate=indices[i],
                    exclusive_support=support.tolist(), proposed_candidate=target,
                    target_proposal_count=target_counts.get(target,0), before_mass=before_mass,
                    retained_mass=float(weights[i])))
        matrix = None
        if isinstance(entry.payload, LightSnapshot):
            matrix = np.zeros((len(original),len(entry.payload.candidates)+1),float)
            matrix[:,0] = 1.
            for i, index in enumerate(indices):
                if index is not None and original[i+1] > 0:
                    fraction = weights[i]/original[i+1]
                    matrix[i+1,index+1] = fraction
                    matrix[i+1,0] = 1.-fraction
            matrix = matrix.tolist()
        audit.append(dict(kind='adjacent_image_transport', frame_id=frame.frame_id,
            source_step=frame.source_step, capture_sim_s=frame.capture_sim_s,
            available_mono_ns=entry.available_mono_ns, jpeg_sha256=entry.jpeg_sha256,
            camera_pose=frame.pose, flow=flow_rows, mapping=mapping,
            capture_to_current_matrix=matrix,
            image_age_at_delivery_s=(delivered_mono_ns-frame.capture_mono_ns)/1e9,
            current_mask_confirmation=isinstance(entry.payload,LightSnapshot),
            candidate_mass=float(weights.sum()), null_probability=1.-float(weights.sum()),
            total_probability=1.))
        previous_gray = image
    return finish(weights, indices)


@dataclass(frozen=True)
class CaptureEvidence:
    evidence_id: str
    kind: str
    probability: tuple[float, ...]
    available_mono_ns: int
    revision: int
    replaces_evidence_id: str | None


class CaptureEvidenceStore:
    """At most cheap then deep per source; replacement, never evidence product."""
    def __init__(self):
        self._bindings, self._history, self._evidence_ids = {}, {}, {}

    def _key(self, snapshot, camera_id):
        _validate_snapshot(snapshot)
        if not isinstance(camera_id,str) or not camera_id:
            raise ValueError('camera stream identity required')
        key = (camera_id,snapshot.frame.frame_id)
        binding = (_frame_binding(snapshot.frame),snapshot.candidate_sha256)
        if key in self._bindings and self._bindings[key] != binding:
            raise ValueError('same source ID has conflicting JPEG/camera/candidate content')
        return key, binding

    def replace_capture_evidence(self, snapshot, probability, *, camera_id, evidence_id,
                                 kind, available_mono_ns):
        key, binding = self._key(snapshot,camera_id)
        values = _probability(probability,len(snapshot.candidates))
        if (not isinstance(evidence_id,str) or not evidence_id or kind not in {'cheap','deep'} or
                type(available_mono_ns) is not int or available_mono_ns<snapshot.frame.capture_mono_ns):
            raise ValueError('source-bound evidence ID, kind and actual availability required')
        signature = (key,kind,values,available_mono_ns)
        if evidence_id in self._evidence_ids:
            if self._evidence_ids[evidence_id] != signature:
                raise ValueError('conflicting duplicate evidence ID')
            return next(record for record in self._history[key] if record.evidence_id == evidence_id)
        history = self._history.get(key,[])
        if history and (kind != 'deep' or history[-1].kind != 'cheap' or
                        available_mono_ns < history[-1].available_mono_ns):
            raise ValueError('no repeated deep, downgrade or reversed replacement')
        if not history and kind != 'cheap':
            raise ValueError('register capture cheap evidence before deep replacement')
        record = CaptureEvidence(evidence_id,kind,values,available_mono_ns,len(history),
                                 history[-1].evidence_id if history else None)
        self._bindings[key] = binding
        self._history.setdefault(key,[]).append(record)
        self._evidence_ids[evidence_id] = signature
        return record

    def as_of(self, snapshot, *, camera_id, delivered_mono_ns):
        key, _ = self._key(snapshot,camera_id)
        if type(delivered_mono_ns) is not int:
            raise ValueError('integer actual delivery required')
        eligible = [record for record in self._history.get(key,[])
                    if record.available_mono_ns<=delivered_mono_ns]
        return eligible[-1] if eligible else None
