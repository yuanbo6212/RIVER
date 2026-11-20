"""Sensor calibration only: no evaluator state and no tracking decisions."""
import math
import numpy as np


def intrinsic_from_attributes(attributes):
    width = int(attributes['image_size_x'])
    height = int(attributes['image_size_y'])
    fov = float(attributes['fov'])
    if width <= 0 or height <= 0 or not math.isfinite(fov) or not 0 < fov < 180:
        raise ValueError('Invalid physical camera attributes')
    focal = width / (2. * math.tan(math.radians(fov) / 2.))
    return np.array([[focal, 0., width/2.], [0., focal, height/2.], [0., 0., 1.]])


def initial_sensor_pose(geom, camera_height):
    # Match the established RGB/instance spawn formula, without asking a newly
    # spawned actor for its still-stale cached transform.
    initial, forward = geom['initial'], geom['forward']
    return dict(x=initial.x-18.*forward.x, y=initial.y-18.*forward.y,
                z=initial.z-18.*forward.z+camera_height,
                pitch=-math.degrees(math.atan2(camera_height,18.)),
                yaw=geom['yaw_rotation'].yaw, roll=0.)
