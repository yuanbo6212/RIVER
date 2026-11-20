"""Z-up pose algebra; heading preserves the source's x/y/yaw convention.

CosFly uses CARLA's forward/right/up convention. These functions do not perform
a handedness conversion; a controller using a different frame must convert it.
"""
import numpy as np


def project_from_carla_pose(world_xyz, pose, intrinsic):
    """Project from CARLA pose angles (degrees); return pixels and signed depth.

    Local axes are forward/right/up, while optical axes are right/down/forward.
    Callers must mask non-positive depths. No stored extrinsic convention is assumed.
    """
    pitch, yaw, roll = np.deg2rad([pose[k] for k in ('pitch', 'yaw', 'roll')])
    cp, sp, cy, sy, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw), np.cos(roll), np.sin(roll)
    rotation = np.array([[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                         [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                         [sp, -cp*sr, cp*cr]])
    camera = (np.asarray(world_xyz) - np.array([pose[k] for k in ('x','y','z')])) @ rotation
    optical = camera[..., [1, 2, 0]] * np.array([1, -1, 1])
    homogeneous = optical @ np.asarray(intrinsic).T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = homogeneous[..., :2] / homogeneous[..., 2:3]
    return uv, optical[..., 2]


def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi


def world_to_heading(vector, yaw):
    v = np.asarray(vector, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack((c * v[..., 0] + s * v[..., 1],
                     -s * v[..., 0] + c * v[..., 1], v[..., 2]), axis=-1)


def heading_to_world(vector, yaw):
    return world_to_heading(vector, -np.asarray(yaw))


def poses_to_actions(positions, yaws, anchor_yaw):
    """Successive deltas, all expressed in the decision-time heading frame.

    N+1 timestamped poses produce N actions. Never rotate each future step
    with a different yaw: that would define a different action space.
    """
    positions, yaws = np.asarray(positions), np.asarray(yaws)
    if positions.ndim != 2 or positions.shape[1] != 3 or yaws.shape != (len(positions),):
        raise ValueError("Expected positions [N+1,3] and yaws [N+1]")
    xyz = world_to_heading(np.diff(positions, axis=0), anchor_yaw)
    return np.column_stack((xyz, wrap_angle(np.diff(yaws))))


def actions_to_poses(actions, position, yaw):
    a = np.asarray(actions)
    xyz = np.asarray(position) + np.cumsum(heading_to_world(a[:, :3], yaw), axis=0)
    angles = wrap_angle(yaw + np.cumsum(a[:, 3]))
    return xyz, angles
