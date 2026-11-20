"""First-action kinematic CARLA camera command; not an aircraft dynamics model."""
import numpy as np
from .geometry import actions_to_poses, world_to_heading, wrap_angle


def online_state(previous_pose, pose, scale):
    """Build the same normalized 32-D ego state used in CosFly training."""
    scale = np.asarray(scale, dtype=float)
    required = ("x", "y", "z", "yaw", "pitch", "roll")
    if (scale.shape != (5,) or np.any(scale <= 0) or not np.isfinite(scale).all()
            or not all(k in pose for k in required)
            or not np.isfinite([pose[k] for k in required]).all()):
        raise ValueError("Invalid pose or normalization scale")
    state = np.zeros(32, np.float32)
    if previous_pose is not None:
        if (not all(k in previous_pose for k in required)
                or not np.isfinite([previous_pose[k] for k in required]).all()):
            raise ValueError("Invalid previous pose")
        delta = np.asarray([pose[k] - previous_pose[k] for k in ("x", "y", "z")])
        yaw = np.deg2rad(pose["yaw"])
        state[:3] = world_to_heading(delta, yaw) / scale[:3]
        state[3] = wrap_angle(yaw - np.deg2rad(previous_pose["yaw"])) / scale[3]
    state[4:6] = [np.deg2rad(pose["pitch"]) / np.pi,
                  np.deg2rad(pose["roll"]) / np.pi]
    return np.clip(state, -1., 1.)


def first_pose_command(normalized_chunk,scale,pose,dt=.5):
    chunk=np.asarray(normalized_chunk,dtype=float)
    scale=np.asarray(scale,dtype=float)
    required=('x','y','z','yaw','pitch','roll')
    if (chunk.ndim!=2 or chunk.shape[0]<1 or chunk.shape[1] not in (5,32)
            or scale.shape!=(5,) or np.any(scale<=0) or not np.isfinite(scale).all()
            or not np.isfinite(chunk).all() or not np.isfinite(dt) or dt<=0
            or not all(k in pose for k in required)
            or not np.isfinite([pose[k] for k in required]).all()):
        raise ValueError('Invalid action, scale, pose or interval')
    if abs(pose['roll'])>1e-3:
        raise ValueError('Only zero-roll camera poses supported')
    physical=chunk[:1,:5]*scale
    xyz,yaw=actions_to_poses(physical[:,:4],[pose[k] for k in ('x','y','z')],np.deg2rad(pose['yaw']))
    pitch=np.deg2rad(pose['pitch'])+np.cumsum(physical[:,4])
    if abs(pitch[0])>np.pi/2:
        raise ValueError('Predicted camera pitch exceeds supported range')
    return dict(x=float(xyz[0,0]),y=float(xyz[0,1]),z=float(xyz[0,2]),
                yaw=float(np.rad2deg(yaw[0])),pitch=float(np.rad2deg(pitch[0])),roll=0.,dt=float(dt))


def chunk_endpoint_pose(normalized_chunk, scale, pose):
    """Decode a complete π0.5 action chunk to its kinematic endpoint."""
    chunk = np.asarray(normalized_chunk, dtype=float)
    scale = np.asarray(scale, dtype=float)
    required = ("x", "y", "z", "yaw", "pitch", "roll")
    if (chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] not in (5, 32)
            or scale.shape != (5,) or np.any(scale <= 0)
            or not np.isfinite(chunk).all() or not np.isfinite(scale).all()
            or not all(k in pose for k in required)
            or not np.isfinite([pose[k] for k in required]).all()
            or abs(pose["roll"]) > 1e-3):
        raise ValueError("Invalid action chunk, scale, or camera pose")
    physical = chunk[:, :5] * scale
    xyz, yaw = actions_to_poses(physical[:, :4],
                                [pose[k] for k in ("x", "y", "z")],
                                np.deg2rad(pose["yaw"]))
    pitch = np.deg2rad(pose["pitch"]) + np.cumsum(physical[:, 4])
    if np.any(np.abs(pitch) > np.pi / 2):
        raise ValueError("Predicted camera pitch exceeds supported range")
    return {"x": float(xyz[-1, 0]), "y": float(xyz[-1, 1]), "z": float(xyz[-1, 2]),
            "yaw": float(np.rad2deg(yaw[-1])), "pitch": float(np.rad2deg(pitch[-1])),
            "roll": 0.0}


def carla_transform(command,carla_module):
    """Construct only; caller decides when/how to apply in a synchronous simulator."""
    return carla_module.Transform(carla_module.Location(**{k:command[k] for k in ('x','y','z')}),
        carla_module.Rotation(**{k:command[k] for k in ('pitch','yaw','roll')}))
