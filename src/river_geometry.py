"""Explicit pinhole geometry for causal RIVER observations and view hypotheses.

The optical basis is x-right, y-down, z-forward. The caller supplies calibrated
intrinsics and an explicit optical-to-world basis. No implicit CARLA Euler,
ground-plane, target depth, detection probability, or occlusion model is used.
All probabilities below are finite-particle *frustum* mass, not visibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from river_belief import ParticleBelief, predict_cv


def _array(value, shape, name):
    out = np.array(value, dtype=np.float64, copy=True)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    out.setflags(write=False)
    return out


def _points(value, dimension, name):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != dimension or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite N x {dimension}")
    return out


@dataclass(frozen=True)
class PinholeCalibration:
    """Rectified pinhole K; pixel center convention is explicit in cx, cy.

    K permits skew, but not lens distortion. Width/height describe the actual
    observation array after any crop/resize, not the encoder's internal square.
    """

    width: int
    height: int
    matrix: np.ndarray
    calibration_id: str

    def __post_init__(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        k = _array(self.matrix, (3, 3), "intrinsic matrix")
        if not np.allclose(k[2], [0., 0., 1.], atol=1e-12, rtol=0.) or abs(k[1, 0]) > 1e-12:
            raise ValueError("K must have rows [fx,skew,cx], [0,fy,cy], [0,0,1]")
        if k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("focal lengths must be positive")
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise ValueError("explicit calibration_id required")
        object.__setattr__(self, "matrix", k)


def calibration_from_horizontal_fov(width: int, height: int, horizontal_fov_deg: float,
                                    *, calibration_id: str):
    """CARLA rectified square-pixel calibration at its original image resolution.

    This matches source_reference/calibrated_view_actions_v5.py:focal_length.
    The principal point is (width/2,height/2), not (width-1)/2. A crop/resize
    requires correspondingly transformed K and must not reuse this constructor.
    """
    if not np.isfinite(horizontal_fov_deg) or not 0 < horizontal_fov_deg < 180:
        raise ValueError("horizontal FOV must be finite and between 0 and 180 degrees")
    if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, (int, np.integer)) or height < 1:
        raise ValueError("height must be a positive integer")
    focal = width / (2 * np.tan(np.deg2rad(horizontal_fov_deg) / 2))
    return PinholeCalibration(width, height,
                              [[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]],
                              calibration_id)


@dataclass(frozen=True)
class CameraFrame:
    """Optical-to-world basis and center at a stated physical time.

    World handedness is explicit: an optical-right-handed basis expressed in a
    left-handed world has determinant -1 (CARLA's x-forward/y-right/z-up case).
    This is a coordinate basis change, not a valid SO(3) rotation in the same
    handedness. Rejecting det=-1 unconditionally would silently break CARLA.
    The archive adapter must supply a camera pose valid at frame capture time.
    A future-view hypothesis supplies its prediction time instead.
    """

    center_world: np.ndarray
    world_from_optical: np.ndarray
    timestamp: float
    world_frame_id: str
    world_handedness: str

    def __post_init__(self):
        center = _array(self.center_world, (3,), "camera center")
        basis = _array(self.world_from_optical, (3, 3), "camera basis")
        if self.world_handedness not in {"right", "left"}:
            raise ValueError("world_handedness must be explicitly right or left")
        determinant = 1. if self.world_handedness == "right" else -1.
        if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-9, rtol=0.) or not np.isclose(
            np.linalg.det(basis), determinant, atol=1e-9, rtol=0.
        ):
            raise ValueError("basis is not orthonormal with the declared handedness")
        if not isinstance(self.world_frame_id, str) or not self.world_frame_id.strip():
            raise ValueError("explicit world_frame_id required")
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        object.__setattr__(self, "center_world", center)
        object.__setattr__(self, "world_from_optical", basis)


def camera_frame_from_carla_pose(pose, *, timestamp: float, world_frame_id: str):
    """Convert the verified original CARLA six-field degree pose convention.

    Euler basis matches source_reference/causal_identity_view_v4.py:rotation;
    camera-local axes there are forward/right/up. Optical axes here are
    right/down/forward, requiring an explicit det=-1 basis conversion. Pose
    must be measured at capture time, or explicitly predicted for a future view.
    Reject extra fields rather than accidentally treating evaluator payloads as
    camera metadata. No target pose or simulator oracle is consumed.
    """
    fields = {"x", "y", "z", "pitch", "yaw", "roll"}
    if not hasattr(pose, "keys") or set(pose) != fields:
        raise ValueError("camera pose must contain exactly x,y,z,pitch,yaw,roll")
    values = np.asarray([pose[k] for k in ("x", "y", "z", "pitch", "yaw", "roll")], float)
    if not np.isfinite(values).all():
        raise ValueError("camera pose must be finite")
    p, y, r = np.deg2rad(values[3:])
    cp, sp, cy, sy, cr, sr = np.cos(p), np.sin(p), np.cos(y), np.sin(y), np.cos(r), np.sin(r)
    world_from_carla_camera = np.array([
        [cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
        [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
        [sp, -cp*sr, cp*cr],
    ])
    carla_camera_from_optical = np.array([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]])
    return CameraFrame(values[:3], world_from_carla_camera @ carla_camera_from_optical,
                       timestamp, world_frame_id, "left")


@dataclass(frozen=True)
class Projection:
    pixels: np.ndarray
    optical_depth: np.ndarray
    in_front: np.ndarray
    in_image: np.ndarray


def project_world(points_world, camera: CameraFrame, calibration: PinholeCalibration,
                  *, near_clip: float) -> Projection:
    """Project world points; behind/near-plane points have NaN image coordinates.

    Image clipping is explicit and half-open [0,width) x [0,height). Points
    outside the image retain their un-clipped coordinates. A point in-frustum
    may still be fully occluded or undetectable; this function cannot tell.
    """
    points = _points(points_world, 3, "world points")
    if not np.isfinite(near_clip) or near_clip <= 0:
        raise ValueError("near_clip must be finite and strictly positive")
    with np.errstate(over="ignore", invalid="ignore"):
        optical = (points - camera.center_world) @ camera.world_from_optical
    if not np.isfinite(optical).all():
        raise ValueError("world-to-optical transform overflow")
    depth = optical[:, 2].copy()
    front = depth > near_clip
    pixels = np.full((len(points), 2), np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        homogeneous = optical[front] @ calibration.matrix.T
        pixels[front] = homogeneous[:, :2] / depth[front, None]
    if not np.isfinite(pixels[front]).all():
        raise ValueError("projection overflow")
    inside = (front & (pixels[:, 0] >= 0) & (pixels[:, 0] < calibration.width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < calibration.height))
    for item in (pixels, depth, front, inside):
        item.setflags(write=False)
    return Projection(pixels, depth, front, inside)


def pixel_world_rays(pixels, camera: CameraFrame, calibration: PinholeCalibration):
    """Return unit world-bearing rays, without guessing target range or motion."""
    pixels = _points(pixels, 2, "pixels")
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    optical = np.linalg.solve(calibration.matrix, homogeneous.T).T
    norms = np.linalg.norm(optical, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("bearing-ray overflow")
    world = (optical / norms[:, None]) @ camera.world_from_optical.T
    world.setflags(write=False)
    return world


def backproject_optical_depth(pixels, depth_z, camera: CameraFrame,
                             calibration: PinholeCalibration):
    """Backproject measured/estimated optical z-depth, NOT Euclidean range.

    A detector box alone provides no depth. The caller must document its causal
    depth sensor or estimation model. Simulator target transforms do not qualify.
    """
    pixels = _points(pixels, 2, "pixels")
    depth = _array(depth_z, (len(pixels),), "optical depths")
    if np.any(depth <= 0):
        raise ValueError("optical depths must be strictly positive")
    optical = np.linalg.solve(
        calibration.matrix, np.column_stack((pixels, np.ones(len(pixels)))).T
    ).T * depth[:, None]
    world = optical @ camera.world_from_optical.T + camera.center_world
    if not np.isfinite(world).all():
        raise ValueError("backprojection overflow")
    world.setflags(write=False)
    return world


def intersect_world_plane(pixels, camera: CameraFrame, calibration: PinholeCalibration,
                          *, plane_point, plane_normal, min_cosine: float = 1e-6):
    """Intersect rays with an explicit causal plane assumption.

    Returns (points, valid). Parallel/backward intersections are NaN and false.
    For ground-plane vehicle reasoning, use a road-contact pixel assumption
    (often box-bottom center), not the visible-mask center. Road slope, truncation
    and occlusion invalidate that assumption; this is not a depth oracle.
    """
    point = _array(plane_point, (3,), "plane point")
    normal = _array(plane_normal, (3,), "plane normal")
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("nonzero plane normal required")
    if not np.isfinite(min_cosine) or not 0 < min_cosine < 1:
        raise ValueError("min_cosine must be strictly between zero and one")
    normal = normal / norm
    rays = pixel_world_rays(pixels, camera, calibration)
    cosine = rays @ normal
    valid = np.abs(cosine) > min_cosine
    distance = np.full(len(rays), np.nan)
    distance[valid] = np.dot(point - camera.center_world, normal) / cosine[valid]
    valid &= np.isfinite(distance) & (distance > 0)
    points = np.full((len(rays), 3), np.nan)
    points[valid] = camera.center_world + rays[valid] * distance[valid, None]
    if not np.isfinite(points[valid]).all():
        raise ValueError("plane-intersection overflow")
    points.setflags(write=False)
    valid.setflags(write=False)
    return points, valid


def reproject_moving_pixels(pixels, depth_z, velocities_world, source: CameraFrame,
                            destination: CameraFrame, calibration: PinholeCalibration,
                            *, near_clip: float):
    """Propagate an explicit CV world estimate and compensate camera pose change.

    Both frames must use the same calibrated pixel system and world coordinate
    frame. Zero velocity is an explicit *static-world assumption*, not an
    identity model. No past box can move under ego-motion merely by resetting
    its pixel velocity. Unknown depth cannot be supplied as a placeholder.
    """
    if (source.world_frame_id != destination.world_frame_id
            or source.world_handedness != destination.world_handedness):
        raise ValueError("source and destination world frames differ")
    if destination.timestamp < source.timestamp:
        raise ValueError("backward-time reprojection rejected")
    world = backproject_optical_depth(pixels, depth_z, source, calibration)
    velocity = _array(velocities_world, world.shape, "world velocities")
    dt = destination.timestamp - source.timestamp
    with np.errstate(over="ignore", invalid="ignore"):
        moved = world + dt * velocity
    if not np.isfinite(moved).all():
        raise ValueError("motion propagation overflow")
    return project_world(moved, destination, calibration, near_clip=near_clip)


@dataclass(frozen=True)
class ParticleImagePrediction:
    belief: ParticleBelief
    projection: Projection
    frustum_mass: float
    conditional_pixel_mean: Optional[np.ndarray]
    conditional_pixel_covariance: Optional[np.ndarray]


def project_particle_belief(belief: ParticleBelief, camera: CameraFrame,
                            calibration: PinholeCalibration, *, near_clip: float,
                            acceleration_spectral_density=0., rng=None):
    """Propagate 3-D CV particles to pose time, then project every particle.

    The caller is responsible for matching belief/camera world frames, as the
    base ParticleBelief type has no frame identifier. Preserve outside-frustum
    particles in the returned belief; they are not missed-detection updates.
    The mean/covariance condition only on in-image mass and are None if zero.
    """
    if belief.dimension != 3:
        raise ValueError("world projection requires 3-D position/velocity particles")
    if camera.timestamp < belief.timestamp:
        raise ValueError("camera time precedes belief; use capture-time replay first")
    predicted = predict_cv(
        belief, camera.timestamp - belief.timestamp,
        acceleration_spectral_density, rng=rng,
    )
    projection = project_world(predicted.states[:, :3], camera, calibration, near_clip=near_clip)
    selected = projection.in_image
    mass = float(predicted.weights[selected].sum())
    mean = covariance = None
    if mass > 0:
        weights = predicted.weights[selected] / mass
        values = projection.pixels[selected]
        mean = weights @ values
        residual = values - mean
        covariance = residual.T @ (weights[:, None] * residual)
        mean.setflags(write=False)
        covariance.setflags(write=False)
    return ParticleImagePrediction(predicted, projection, mass, mean, covariance)
