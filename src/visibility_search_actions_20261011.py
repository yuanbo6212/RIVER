"""Causal coverage and bounded-search controls; no identity/GT action inputs.

Coverage is projected *observed box* area, not predicted occlusion, confidence,
or information gain. A flat ground/contact-ray approximation is explicit.
The search starts with the existing rightward prior, not an oracle direction.
"""
import math
import numpy as np
from calibrated_view_actions_v5 import focal_length
from causal_identity_view_v4 import rotation, pixel_ray, project_ray

MODES = ('fixed_orbit_right', 'causal_visibility', 'causal_search')


def right_motion(origin, amount=.08):
    a = math.radians(origin['yaw'])
    return np.array([-math.sin(a)*amount, math.cos(a)*amount, 0.])


def observed_boxes(candidates, width, height):
    boxes = []
    for c in candidates:
        b = np.asarray(c['xyxy'], float)
        if b.shape != (4,) or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
            raise ValueError('invalid current-frame candidate box')
        boxes.append(b.tolist())
    return boxes


def coverage_pointing(boxes, pose, width, height, fov):
    """Center all current candidate angular bounds, without a selected ID."""
    f = focal_length(width, height, fov)
    if not boxes:
        return -1., 0.
    xs = [math.atan2(b[k]-width/2., f) for b in boxes for k in (0, 2)]
    ys = [math.atan2(-(b[k]-height/2.), f) for b in boxes for k in (1, 3)]
    return (float(np.clip(math.degrees((min(xs)+max(xs))/2), -2., 2.)),
            float(np.clip(math.degrees((min(ys)+max(ys))/2), -1., 1.)))


def box_world_billboard(box, pose, origin, width, height, fov):
    """Reproject an observed box at its foot-ray depth; not an amodal mesh."""
    r = rotation(pose)
    foot = r @ pixel_ray((box[0]+box[2])/2., box[3], width, height, fov)
    plane_z = origin['z'] - 5.
    if foot[2] >= -1e-7:
        return None
    t = (plane_z-pose['z'])/foot[2]
    if not np.isfinite(t) or t <= 0:
        return None
    camera = np.array([pose[k] for k in ('x', 'y', 'z')])
    return [camera + t*(r @ pixel_ray(u, v, width, height, fov))
            for u in (box[0], box[2]) for v in (box[1], box[3])]


def projected_coverage(points, proposed, width, height, fov):
    """Lexicographic: worst observed-box retention, then total pixel area."""
    camera = np.array([proposed[k] for k in ('x', 'y', 'z')])
    inverse = rotation(proposed).T
    retained, areas = [], []
    for corners in points:
        pixels = [project_ray(inverse @ (p-camera), width, height, fov) for p in corners]
        if any(p is None for p in pixels):
            retained.append(0.); areas.append(0.); continue
        x0, y0 = min(p[0] for p in pixels), min(p[1] for p in pixels)
        x1, y1 = max(p[0] for p in pixels), max(p[1] for p in pixels)
        area = max(0., x1-x0)*max(0., y1-y0)
        clipped = max(0., min(width,x1)-max(0.,x0))*max(0.,min(height,y1)-max(0.,y0))
        retained.append(clipped/area if area > 0 else 0.)
        areas.append(clipped/(width*height))
    return (min(retained) if retained else 0., sum(areas))


def search_yaw_target(decision_index):
    # Deterministic angular sweep; no future detections or hidden-swap timing.
    if decision_index < 8:
        return -15.*(decision_index+1)/8.
    if decision_index < 20:
        return -15.+30.*(decision_index-7)/12.
    return 15.-15.*(decision_index-19)/9.


def action_residual(mode, decision_index, candidates, selected_index, identity,
                    pose, origin, width, height, fov):
    if mode not in MODES or not 0 <= decision_index < 29:
        raise ValueError('unknown control or decision clock')
    # Fixed arm is bit-for-bit the old strong right-orbit residual.
    if mode == 'fixed_orbit_right':
        d = right_motion(origin)
        return dict(x=float(d[0]),y=float(d[1]),z=0.,yaw=-1.,pitch=0.), dict(
            branch=mode, identity_state_consumed=False)
    boxes = observed_boxes(candidates, width, height)
    yaw, pitch = coverage_pointing(boxes, pose, width, height, fov)
    branch = dict(branch=mode, identity_state_consumed=False,
                  candidate_observation_consumed=True, candidate_count=len(boxes),
                  selected_identity_ignored=True, score_is_information_gain=False)
    if mode == 'causal_search':
        d = right_motion(origin)
        if not boxes:
            target = origin['yaw'] + search_yaw_target(decision_index)
            yaw = float(np.clip((target-pose['yaw']+180)%360-180, -2., 2.))
            branch.update(phase='blind_bounded_angular_sweep',desired_yaw=target)
        else:
            branch.update(phase='current_candidate_coverage_servo')
    elif not boxes:
        d = right_motion(origin)
        branch.update(phase='no_candidates_shared_right_prior')
    else:
        points = [box_world_billboard(b, pose, origin, width, height, fov) for b in boxes]
        if any(p is None for p in points):
            d = right_motion(origin)
            branch.update(phase='invalid_ground_projection_shared_right_prior')
        else:
            right = right_motion(origin)
            a = math.radians(origin['yaw'])
            forward = np.array([.08*math.cos(a), .08*math.sin(a), 0.])
            options = [('right',right),('left',-right),('forward',forward),
                       ('backward',-forward),('rise',np.array([0.,0.,.08])),
                       ('hold',np.zeros(3))]
            scores = []
            for name, move in options:
                q = dict(pose)
                for k, v in zip(('x','y','z'), move): q[k] += float(v)
                q['yaw'] += yaw; q['pitch'] += pitch
                scores.append(projected_coverage(points, q, width, height, fov))
            choice = max(range(len(options)), key=lambda i: scores[i])
            d = options[choice][1]
            branch.update(phase='receding_observed_box_coverage', chosen_motion=options[choice][0],
                          motion_options=[p[0] for p in options], projection_scores=[list(s) for s in scores],
                          projection_uses='current_observed_bbox_at_flat_ground_foot_ray_depth',
                          assumed_initial_mounting_height_m=5., predicts_hidden_occlusion=False)
    return dict(x=float(d[0]),y=float(d[1]),z=float(d[2]),yaw=yaw,pitch=pitch), branch
