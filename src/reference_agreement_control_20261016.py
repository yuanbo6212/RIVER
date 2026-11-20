"""Bounded development: preserve reference control, test one neural feedback guard.

Agreement is a current-observation consistency check, never proof of identity.
No new detector, template update, GT depth, calibration fit, or hidden-state input.
"""
import numpy as np
from calibrated_route_adapter_20261014 import CalibratedRoute, configure
from live_neural_route_20261016 import current_observation
from visibility_search_actions_20261011 import coverage_pointing, right_motion

METHODS = ('reference_non_goal', 'reference_search_left', 'agreement_guard_non_goal')
AGREEMENT_IOU = .5


def overlap_xywh_xyxy(a, b):
    if a is None or b is None:
        return 0.
    intersection = max(0., min(a[0]+a[2], b[2])-max(a[0], b[0])) * max(
        0., min(a[1]+a[3], b[3])-max(a[1], b[1]))
    union = a[2]*a[3] + (b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


class ReferenceAgreementRoute(CalibratedRoute):
    def residual(self, index, pose, candidates, selected, identity):
        if self.neural_observation['step'] != 185+index:
            raise ValueError('Cached/future observation forbidden')
        move, branch = super().residual(index, pose, candidates, selected, identity)
        score = branch['appearance_difference']
        box = self.neural_observation['box_xywh']
        iou = overlap_xywh_xyxy(box, candidates[selected]['xyxy']) if selected is not None else 0.
        allowed = box is not None and score is not None and score > 0. and iou >= AGREEMENT_IOU
        used = self.study_method == 'agreement_guard_non_goal' and allowed
        if used:
            x, y, w, h = box
            move['yaw'], move['pitch'] = coverage_pointing(
                [[x, y, x+w, y+h]], pose, self.width, self.height, self.fov)
            branch['phase'] = 'current_reference_agreed_neural_pointing'
        if self.study_method == 'reference_search_left':
            vector = -right_motion(self.origin)
            move.update(x=float(vector[0]), y=float(vector[1]), z=0.)
            branch.update(translation_candidate_indices=[], translation_bearings_from_origin_deg=[],
                translation_direction=-1., translation_direction_changed=False,
                translation_identity_conditioned=False, fixed_left_translation=True)
        branch.update(branch=self.study_method,
            reference_pointing_preserved_unless_current_agreement=not used,
            reference_neural_iou=iou, agreement_iou_threshold=AGREEMENT_IOU,
            current_observation_agreement=allowed, neural_pointing_actually_used=used,
            agreement_not_identity_guarantee=True, raw_neural_readout_not_modified=True)
        return move, branch


class ReferenceAgreementControl:
    def __init__(self, url, token, scale, mode, origin, sensor_attributes, infer=None):
        if mode not in METHODS:
            raise ValueError('Unregistered development condition')
        self.impl = ReferenceAgreementRoute(url, token, scale, 'non_goal_away_route', origin, infer)
        configure(self.impl, sensor_attributes)
        self.impl.study_method = mode
        self.mode, self.next_step = mode, 0

    @property
    def path_length(self):
        return self.impl.path_length

    def observe(self, jpeg, step, pose, initial_box, candidates, selected, identity, neural_reply):
        if step != self.next_step:
            raise ValueError('Unordered observation')
        obs = current_observation(neural_reply, jpeg, step)
        self.impl.neural_observation = obs
        command, trace = self.impl.observe(jpeg, step, pose, initial_box, candidates, selected, identity)
        self.next_step += 1
        if command is not None:
            trace = dict(trace, reason=self.mode, current_neural_observation=obs,
                neural_recovery_consumed_by_pointing=trace['branch']['neural_pointing_actually_used'],
                native_neural_pid_executed=False, shared_dual_observer_stack=True,
                action_camera_intrinsics=dict(self.impl.intrinsics),
                calibration_source='actual_sensor_attributes',
                current_agreement_not_identity_safety_guarantee=True)
        return command, trace
