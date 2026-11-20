"""Fixed-command E02 branch control and causal auxiliary identity adapter.

This is a finite development adapter, not a trained risk planner, physical
identity-state posterior, flight-dynamics controller, or wall-clock scheduler.
The three arms differ only in assigned lateral residual. All share the same
current-evidence pointing rule and 29 actual pi0.5 requests with zero execution.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from river_calibration import HistogramSpec
from river_set_association import RepresentativeAssociationModel


MODEL_FILE_SHA256 = "3a008dd7bf07d7d648f4b2947b22a5aa8b87b781b75b10ccebe8f2b7a59aa986"
MODEL_FINGERPRINT = "288db070a88e7f7d413a38be69b959b661bdf885d9346bd74f471e27798af209"
POSE_FIELDS = ("x", "y", "z", "pitch", "yaw", "roll")
MODES = {"e02_branch_left": -1., "e02_branch_stay": 0., "e02_branch_right": 1.}
IDENTITY_MODE = "river_auxiliary_representative_v2"
ACTION_STEPS = tuple(range(185, 214))
MAX_STEP_M = 3. / 29.
LATERAL_STEP_M = .08


def _pose(pose):
    if set(pose) != set(POSE_FIELDS) or not np.isfinite([pose[k] for k in POSE_FIELDS]).all():
        raise ValueError("camera pose requires finite six fields only")
    if abs(pose["roll"]) > 1e-3:
        raise ValueError("zero-roll camera contract required")
    return {k: float(pose[k]) for k in POSE_FIELDS}


def _box(box, width, height, *, initial=False):
    b = np.asarray(box, float)
    if b.shape != (4,) or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("invalid observed rectangle")
    if initial and not (0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height):
        raise ValueError("initial task rectangle outside image")
    return b.tolist()


def _candidates(candidates, width, height):
    for c in candidates:
        if set(c) != {"xyxy", "confidence", "class_id", "mask"}:
            raise ValueError("candidate must contain only current observed fields")
        _box(c["xyxy"], width, height)
        if not math.isfinite(float(c["confidence"])) or not 0 <= c["confidence"] <= 1:
            raise ValueError("invalid detector confidence")
        mask = np.asarray(c["mask"])
        if mask.shape != (height, width) or not np.isin(mask, [0, 1]).all():
            raise ValueError("invalid current candidate mask")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_development_representative(path, expected_sha256=MODEL_FILE_SHA256):
    """Load exactly the corrected frozen auxiliary model; never fit on import."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("wrong/corrupted frozen E01 corrected model file")
    a = json.loads(raw)
    if a["feature_names"] != ["positive_similarity", "known_negative_similarity"]:
        raise ValueError("wrong auxiliary score model")
    metadata = dict(a["metadata"])
    fingerprint = metadata.pop("model_sha256")
    content = dict(metadata=metadata, features=a["feature_names"], edges=a["bin_edges"],
                   pseudocount=a["histogram_pseudocount"], observation_definition=a["observation_definition"],
                   target_mass=a["target_bin_mass"], nuisance_mass=a["nuisance_bin_mass"],
                   q=a["candidate_present_prior_q"], rate=a["expected_nuisance_count"])
    if fingerprint != MODEL_FINGERPRINT or _hash(content) != fingerprint:
        raise ValueError("frozen auxiliary model content fingerprint mismatch")
    if not metadata["data_id"].startswith("e01_designated_geometry_export_v2/"):
        raise ValueError("withdrawn physical-role label model prohibited")
    if metadata["policy_release"] is not False:
        raise ValueError("unexpected artifact release metadata")
    metadata["model_sha256"] = fingerprint
    spec = HistogramSpec(tuple(a["feature_names"]), tuple(np.asarray(e) for e in a["bin_edges"]),
                         a["histogram_pseudocount"], a["observation_definition"])
    return RepresentativeAssociationModel(spec, a["target_bin_mass"], a["nuisance_bin_mass"],
        a["candidate_present_prior_q"], a["expected_nuisance_count"], metadata)


class CausalRepresentativeIdentity:
    """Wrap the ORIGINAL fixed task-HSV extractor, replacing only its selection.

    The artifact was never released as a physical identity posterior. Explicit
    allow_development_use opts into this bounded new E02 development trial; the
    original policy_release flag is not changed. No task-role integer is accepted.
    """
    def __init__(self, mode="task_anchored_hsv", *, model_path=None, extractor=None,
                 allow_development_use=False, expected_model_file_sha256=MODEL_FILE_SHA256):
        if not allow_development_use:
            raise ValueError("explicit bounded development use required")
        if mode != "task_anchored_hsv":
            raise ValueError("only frozen original task-HSV references supported")
        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "artifacts/e01_designated_set_association_v2/model.json"
        self.model = load_development_representative(model_path, expected_model_file_sha256)
        if extractor is None:
            from task_anchored_identity_20261013 import TaskAnchoredIdentity
            extractor = TaskAnchoredIdentity("task_anchored_hsv")
        self.extractor = extractor

    @property
    def step(self):
        return self.extractor.step

    def update(self, rgb, candidates, step, pose, initial_box=None, fov=90.):
        if type(step) is not int or step != self.step+1 or ((step == 0) != (initial_box is not None)):
            raise ValueError("unordered current frame or later task annotation")
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("uint8 RGB observation required")
        _pose(pose)
        _candidates(candidates, image.shape[1], image.shape[0])
        selected, detail = self.extractor.update(image, candidates, step, pose, initial_box, fov)
        if self.step != step or detail["gt_inputs"] or detail["future_frames_available"]:
            raise ValueError("extractor violates causal input clock")
        if detail["selected_index"] != selected or detail.get("template_updated") or detail.get("negative_memory_updated"):
            raise ValueError("original frozen reference contract changed")
        scores = detail["candidate_scores"]
        indices = [s["index"] for s in scores]
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(candidates) for i in indices):
            raise ValueError("invalid original score indices")
        raw_selected = selected
        probabilities = None
        if step == 0:
            # Legal task-box bootstrap is a separate information condition.
            reason = "legal_initial_task_box_bootstrap"
        elif set(indices) != set(range(len(candidates))):
            # The fit population had complete score coverage. Retain all new
            # proposals and abstain if any are unscored; do not hide a domain gap
            # by renormalizing a subset or invent a confident null probability.
            selected = None
            reason = "unscored_current_proposal_unknown"
        elif detail["negative_reference_count"] == 0 and candidates:
            selected = None
            reason = "missing_frozen_negative_reference_unknown"
        else:
            lookup = {s["index"]: s for s in scores}
            values = np.array([[lookup[j]["positive_similarity"], lookup[j]["known_negative_similarity"]]
                               for j in range(len(candidates))], float).reshape(-1, 2)
            probabilities = self.model.predict(values)
            winner = int(np.argmax(probabilities))  # null first on exact ties
            selected = winner - 1 if winner else None
            reason = "auxiliary_representative_argmax" if winner else "auxiliary_null_argmax"
            scores = [dict(s, representative_probability=float(probabilities[s["index"]+1])) for s in scores]
        return selected, dict(detail, mode=IDENTITY_MODE, reason=reason, selected_index=selected,
            candidate_scores=scores, raw_positive_selector_index=raw_selected,
            auxiliary_probabilities=probabilities.tolist() if probabilities is not None else None,
            auxiliary_model_fingerprint=self.model.metadata["model_sha256"],
            probability_scope="auxiliary_uniform_representative_not_physical_membership",
            development_trial=True, artifact_policy_release=False,
            unscored_candidate_count=len(candidates)-len(indices),
            calibrated_identity_probability=False)


class E02BranchControl:
    """Observe-compatible fixed-command controller, same budget upper bound.

    Stay has zero lateral residual, so equal permitted motion budget does not
    mean equal realized distance. Pointing is shared *feedback*, not equal
    future yaw commands under different future images.
    """
    def __init__(self, url, token, scale, mode, origin, sensor_attributes, infer=None,
                 prior_enabled=False, *, state_builder=None, pose_decoder=None):
        if mode not in MODES or prior_enabled is not False:
            raise ValueError("finite three-action, zero-executed-prior pilot only")
        if set(sensor_attributes) != {"image_size_x", "image_size_y", "fov"}:
            raise ValueError("actual camera calibration required")
        width, height, fov = (float(sensor_attributes[k]) for k in ("image_size_x", "image_size_y", "fov"))
        if not np.isfinite([width, height, fov]).all() or width < 1 or height < 1 or width != int(width) or height != int(height) or not 0 < fov < 180:
            raise ValueError("invalid actual sensor calibration")
        self.width, self.height, self.fov = int(width), int(height), fov
        self.focal = width / (2 * math.tan(math.radians(fov)/2))
        self.origin = _pose(origin)
        self.scale = np.asarray(scale, float).copy()
        if self.scale.shape != (5,) or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("original positive five-axis normalization required")
        if state_builder is None or pose_decoder is None:
            from cosfly_pi05.control import online_state, first_pose_command
            state_builder = state_builder or online_state
            pose_decoder = pose_decoder or first_pose_command
        self.state_builder, self.pose_decoder = state_builder, pose_decoder
        self.url, self.token, self.infer = url.rstrip("/"), token, infer
        self.mode, self.direction = mode, MODES[mode]
        self.history, self.reference, self.initial_box, self.previous_pose = [], None, None, None
        self.path_length = 0.
        self.call_count = 0

    def _request(self, payload):
        if self.infer is not None:
            return self.infer(payload)
        request = Request(self.url+"/infer", data=json.dumps(payload).encode(),
                          headers={"Content-Type":"application/json", "Authorization":"Bearer "+self.token})
        with urlopen(request, timeout=180) as response:
            return json.load(response)

    def _point(self, index, pose, candidates, selected):
        if selected is not None:
            b = candidates[selected]["xyxy"]
            xs = [math.atan2(b[k]-self.width/2, self.focal) for k in (0,2)]
            ys = [math.atan2(-(b[k]-self.height/2), self.focal) for k in (1,3)]
            yaw = float(np.clip(math.degrees(sum(xs)/2), -2., 2.))
            pitch = float(np.clip(math.degrees(sum(ys)/2), -1., 1.))
            return yaw, pitch, "current_auxiliary_candidate_pointing"
        if index < 8:
            sweep = -15.*(index+1)/8.
        elif index < 20:
            sweep = -15.+30.*(index-7)/12.
        else:
            sweep = 15.-15.*(index-19)/9.
        change = (self.origin["yaw"]+sweep-pose["yaw"]+180.) % 360.-180.
        return float(np.clip(change, -2., 2.)), 0., "common_causal_angular_sweep"

    def observe(self, jpeg, step, pose, initial_box, candidates, selected_index, identity, neural_reply=None):
        pose = _pose(pose)
        if type(step) is not int or not 0 <= step < 230 or step != len(self.history) or ((step == 0) != (initial_box is not None)):
            raise ValueError("unordered RGB or later task annotation")
        if not isinstance(jpeg, bytes) or not jpeg:
            raise ValueError("current immutable encoded RGB required")
        _candidates(candidates, self.width, self.height)
        if identity["step"] != step or identity["selected_index"] != selected_index or identity["gt_inputs"] or identity["future_frames_available"]:
            raise ValueError("identity/readout input is noncausal")
        if set(identity) & {"task", "depth_swap", "exit_mode", "target_box", "distractor_box", "target_px", "distractor_px", "actor_id", "target_pose"}:
            raise ValueError("world/evaluator-only information in identity input")
        if identity["mode"] != IDENTITY_MODE or identity["probability_scope"] != "auxiliary_uniform_representative_not_physical_membership":
            raise ValueError("wrong identity readout for fixed E02 trial")
        if selected_index is not None and (type(selected_index) is not int or not 0 <= selected_index < len(candidates)):
            raise ValueError("invalid current selected proposal")
        if neural_reply is not None and (neural_reply.get("step") != step or neural_reply.get("delivered_prefix") != step+1):
            raise ValueError("native dual-observer response is stale")
        self.history.append(jpeg)
        if step == 0:
            self.reference, self.initial_box = jpeg, _box(initial_box,self.width,self.height,initial=True)
        if step not in ACTION_STEPS:
            self.previous_pose = pose
            return None, dict(reason="shared_fixed_camera_phase" if step < 185 else "fixed_deadline_observation",
                              rgb_prefix=step+1, gt_inputs=False, auxiliary_identity_readout_unchanged=True)
        state = np.asarray(self.state_builder(self.previous_pose,pose,self.scale), float)
        if state.shape != (32,) or not np.isfinite(state).all():
            raise ValueError("original normalized state builder violated shape")
        seed = 2026101300+step
        payload = dict(images_b64=[base64.b64encode(x).decode("ascii") for x in reversed(self.history[-5:])],
            reference_b64=base64.b64encode(self.reference).decode("ascii"), reference_box_xyxy=self.initial_box,
            state=state.tolist(), seed=seed)
        request_start_ns = time.monotonic_ns()
        real = np.asarray(self._request(payload)["normalized_actions"], float)
        request_end_ns = time.monotonic_ns()
        self.call_count += 1
        if real.shape != (8,5) or not np.isfinite(real).all():
            raise ValueError("invalid actual shared pi0.5 chunk")
        decoded = np.zeros((8,5), float)
        nominal = self.pose_decoder(decoded,self.scale,pose,dt=.1)
        translation_error = max(abs(nominal[k]-pose[k]) for k in ("x","y","z"))
        # The original five-axis decoder explicitly commands zero roll.  A
        # measured sub-millidegree roll (already admitted by _pose) is not a
        # nonzero prior action.  Verify that decoder convention separately;
        # do not alter the measured state or the command calculated below.
        angular_error = max(abs((nominal[k]-pose[k]+180.) % 360.-180.) for k in ("pitch","yaw"))
        decoder_roll_error = abs((nominal["roll"]+180.) % 360.-180.)
        if max(translation_error,angular_error,decoder_roll_error) > 1e-10:
            raise ValueError("original zero-decoded prior is not identity camera command")
        yaw, pitch, phase = self._point(step-185, pose, candidates, selected_index)
        radians = math.radians(self.origin["yaw"])
        residual = dict(x=-math.sin(radians)*LATERAL_STEP_M*self.direction,
                        y=math.cos(radians)*LATERAL_STEP_M*self.direction,z=0.,yaw=yaw,pitch=pitch)
        command = dict(pose)
        delta = np.array([residual[k] for k in ("x","y","z")])
        length = float(np.linalg.norm(delta))
        cap = min(MAX_STEP_M,max(0.,3.-self.path_length))
        if length > cap:
            delta *= cap/length
        for key,value in zip(("x","y","z"),delta):
            command[key] += float(value)
        for key,total_limit,step_limit in (("yaw",15.,2.),("pitch",10.,1.)):
            offset = (pose[key]-self.origin[key]+180.) % 360.-180.
            command[key] = self.origin[key]+float(np.clip(offset+np.clip(residual[key],-step_limit,step_limit),-total_limit,total_limit))
        command["roll"] = 0.
        cost = float(np.linalg.norm(delta))
        self.path_length += cost
        self.previous_pose = pose
        trace = dict(reason=self.mode, assigned_action={-1.:"Left",0.:"Stay",1.:"Right"}[self.direction],
            origin_pose=dict(self.origin), rgb_prefix=step+1, seed=seed, pi05_chunk=real.tolist(),
            action_decode_chunk=decoded.tolist(), state=state.tolist(), command=command,
            history_steps_latest_first=list(range(step,step-5,-1)), control_residual=residual,
            branch=dict(branch=self.mode,phase=phase,assigned_lateral_direction=self.direction,
                direction_not_redecided_from_hidden_state=True,identity_state_consumed=True,
                pointing_candidate_indices=[] if selected_index is None else [selected_index],
                mask_readout_changed=False,score_is_information_gain=False),
            command_path_length_m=self.path_length, command_translation_cap_m=MAX_STEP_M,
            step_command_path_m=cost, readout_selected_index=selected_index, identity_state_step=step,
            task_box_source_step=0, gt_inputs=False, prior_consumed_by_action=False,
            prior_model_executed_for_matched_compute=True, execute_at_next_observation=True,
            prior_request_sha256=_hash(payload), shared_prior_call_count=self.call_count,
            prior_request_start_monotonic_ns=request_start_ns,prior_request_end_monotonic_ns=request_end_ns,
            prior_rpc_wall_ms=(request_end_ns-request_start_ns)/1e6,
            native_neural_pointing_ignored=True, fixed_command_not_wallclock_trial=True,
            action_camera_intrinsics=dict(width=self.width,height=self.height,fov=self.fov,fx=self.focal,fy=self.focal))
        return command, trace
