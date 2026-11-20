"""The single v5 correction: use the supplied sensor intrinsics in actions.

Assignment/readout remains v4. The 1.5 m apparent-height assumption remains
an explicitly approximate depth proxy, never GT or measured scene depth.
"""
import math
import numpy as np
from causal_identity_view_v4 import MODES, rotation, pixel_ray, project_ray
from causal_identity_view_v4 import action_residual as original_action_residual


def focal_length(width, height, fov):
    if not np.isfinite([width, height, fov]).all() or width <= 0 or height <= 0 or not 0 < fov < 180:
        raise ValueError('invalid or absent actual camera intrinsics')
    return width / (2 * math.tan(math.radians(fov) / 2))


def follow_residual(candidate, pose, width, height, fov):
    f = focal_length(width, height, fov)
    residual = dict(x=0., y=0., z=0., yaw=0., pitch=0.)
    if candidate is None:
        residual['z'] = .08
        return residual
    b = candidate['xyxy']
    dx, dy = (b[0]+b[2])/2-width/2, (b[1]+b[3])/2-height/2
    yaw = math.radians(pose['yaw'])
    lateral = float(np.clip(dx/f*.08, -.08, .08))
    residual.update(x=-math.sin(yaw)*lateral, y=math.cos(yaw)*lateral,
                    yaw=float(np.clip(math.degrees(math.atan2(dx,f)), -2., 2.)),
                    pitch=float(np.clip(-math.degrees(math.atan2(dy,f)), -1., 1.)))
    return residual


def separation_probe(candidates, indices, pose, width, height, fov):
    f = focal_length(width, height, fov)
    yaw = math.radians(pose['yaw'])
    options = [('rise', np.array([0.,0.,.08])),
               ('right', np.array([-.08*math.sin(yaw),.08*math.cos(yaw),0.])),
               ('left', np.array([.08*math.sin(yaw),-.08*math.cos(yaw),0.]))]
    r = rotation(pose)
    centers = []
    for index in indices:
        b = candidates[index]['xyxy']
        d = f*1.5/max(1., b[3]-b[1])
        centers.append(r @ (pixel_ray((b[0]+b[2])/2, (b[1]+b[3])/2, width,height,fov)*d))
    if len(centers) != 2:
        return dict(x=0.,y=0.,z=.08,yaw=0.,pitch=0.), dict(
            chosen='rise', scores=[], reason='insufficient_two_candidate_geometry')
    scores = []
    for name, delta in options:
        p = [project_ray(r.T @ (c-delta), width,height,fov) for c in centers]
        if any(x is None for x in p):
            value = -1e6
        else:
            edge = min(min(x[0],width-x[0],x[1],height-x[1]) for x in p)
            value = float(np.linalg.norm(np.array(p[0])-p[1]))/f-max(0.,20.-edge)/f
        scores.append(dict(action=name, projection_proxy_score=value))
    chosen = max(range(3),key=lambda i:(scores[i]['projection_proxy_score'],-i))
    delta = options[chosen][1]
    return dict(x=float(delta[0]),y=float(delta[1]),z=float(delta[2]),yaw=0.,pitch=0.), dict(
        chosen=options[chosen][0], scores=scores, reason='apparent_size_separation_proxy',
        assumed_object_height_m=1.5, is_measured_depth=False, is_information_gain=False)


def action_residual(mode, decision_index, candidates, selected_index, identity, pose, origin,
                    width, height, fov):
    focal_length(width,height,fov)
    if mode not in MODES or not 0 <= decision_index < 29:
        raise ValueError('unfrozen control or action clock')
    if mode in MODES[:4]:
        return original_action_residual(mode,decision_index,candidates,selected_index,identity,pose,origin)
    candidate = candidates[selected_index] if selected_index is not None else None
    if mode == 'task_visibility' or identity['status'] == 'task_candidate_available':
        return follow_residual(candidate,pose,width,height,fov), dict(
            branch='task_visibility_follow', identity_state_consumed=True,
            selected_candidate_consumed=selected_index)
    alternatives = identity['assignment_alternatives']
    indices = ([alternatives[0]['task_index'],alternatives[0]['context_index']]
               if alternatives and alternatives[0]['context_index'] is not None else [])
    residual,detail = separation_probe(candidates,indices,pose,width,height,fov)
    return residual, dict(branch='identity_evidence_probe',identity_state_consumed=True,
                          identity_status_consumed=identity['status'],projection_proxy=detail)
