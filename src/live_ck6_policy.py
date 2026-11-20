"""RGB/self-pose-only development controls around the same frozen pi0.5 prior.

No evaluator imports, actor coordinates, prescribed target motion or GT masks.
These nominal/fixed controls are engineering baselines, not the main method.
"""
import base64
import json
import math
from urllib.request import Request, urlopen

import numpy as np
from cosfly_pi05.control import first_pose_command, online_state


def bounded_command(pose, proposed, origin, remaining, max_step=0.20):
    delta = np.array([proposed[k] - pose[k] for k in ("x", "y", "z")], dtype=float)
    if not np.isfinite(delta).all() or not np.isfinite(remaining) or remaining < 0:
        raise ValueError("invalid camera command")
    length = float(np.linalg.norm(delta))
    cap = min(max_step, remaining)
    if length > cap:
        delta *= cap / length
    result = dict(pose)
    for k, change in zip(("x", "y", "z"), delta):
        result[k] += float(change)
    for k, total_limit, step_limit in (("yaw", 15., 2.), ("pitch", 10., 1.)):
        change = (proposed[k] - pose[k] + 180.) % 360. - 180.
        offset = (pose[k] - origin[k] + 180.) % 360. - 180.
        result[k] = origin[k] + float(np.clip(offset + np.clip(change, -step_limit, step_limit),
                                             -total_limit, total_limit))
    result["roll"] = 0.0
    return result, float(np.linalg.norm(delta))


class LiveCK6Policy:
    def __init__(self, url, token, scale, mode, origin):
        if mode not in ("nominal", "fixed_orbit"):
            raise ValueError("unfrozen method is not admitted to this engineering test")
        self.url, self.token, self.scale, self.mode = url.rstrip("/"), token, scale, mode
        self.origin = dict(origin)
        self.history = []
        self.reference = None
        self.initial_box = None
        self.previous_pose = None
        self.path_length = 0.0

    def observe(self, jpeg, step, pose, initial_box=None):
        # Frame arrival order is enforced by the caller's causal observer too.
        if step != len(self.history):
            raise ValueError("unordered RGB observation")
        self.history.append(jpeg)
        if step == 0:
            self.reference, self.initial_box = jpeg, list(initial_box)
        # Shared camera stays fixed through all 150+ hidden observations.
        # Frozen experiment clock, never a GT visibility event, starts action.
        if not 180 <= step < 209:
            self.previous_pose = dict(pose)
            return None, {"reason": "shared_fixed_camera_phase", "rgb_prefix": step + 1}
        frames = list(reversed(self.history[-5:]))
        frames.extend([frames[-1]] * (5 - len(frames)))
        state = online_state(self.previous_pose, pose, self.scale)
        seed = 2026100700 + step
        payload = {"images_b64": [base64.b64encode(x).decode("ascii") for x in frames],
                   "reference_b64": base64.b64encode(self.reference).decode("ascii"),
                   "reference_box_xyxy": self.initial_box, "state": state.tolist(), "seed": seed}
        request = Request(self.url + "/infer", data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json",
                                   "Authorization": "Bearer " + self.token})
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        chunk = np.asarray(result["normalized_actions"], dtype=float)
        if chunk.shape != (8, 5) or not np.isfinite(chunk).all():
            raise ValueError("invalid shared pi0.5 action chunk")
        proposed = first_pose_command(chunk, self.scale, pose, dt=.1)
        if self.mode == "fixed_orbit":
            angle = math.radians(self.origin["yaw"])
            proposed["x"] -= .10 * math.sin(angle)
            proposed["y"] += .10 * math.cos(angle)
            proposed["yaw"] -= 1.0
        command, cost = bounded_command(pose, proposed, self.origin, max(0., 3. - self.path_length))
        self.path_length += cost
        record = {"reason": self.mode, "rgb_prefix": step + 1, "seed": seed,
                  "history_steps_latest_first": list(range(step, step - 5, -1)),
                  "pi05_chunk": chunk.tolist(), "state": state.tolist(),
                  "command": command, "command_path_length_m": self.path_length,
                  "task_box_source_step": 0, "gt_inputs": False}
        self.previous_pose = dict(pose)
        return command, record
