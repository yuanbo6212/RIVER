"""Development view policy primitives. No GT, hidden motion, or future RGB.

Shape references and assignment costs are heuristic evidence, not ReID
probabilities. Rotation normalization does not compensate translation. The
optional probe projection uses an explicit 1.5 m upright-object approximation;
it is a view-separation proxy, not a depth measurement or information gain.
"""
import math
import numpy as np

POSE_FIELDS = ('x', 'y', 'z', 'pitch', 'yaw', 'roll')
MODES = ('nominal', 'fixed_rise', 'fixed_orbit', 'simple_search',
         'task_visibility', 'identity_probe')


def rotation(pose):
    if not all(k in pose for k in POSE_FIELDS) or not np.isfinite([pose[k] for k in POSE_FIELDS]).all():
        raise ValueError('invalid ego pose')
    p, y, r = np.deg2rad([pose[k] for k in ('pitch', 'yaw', 'roll')])
    cp, sp, cy, sy, cr, sr = np.cos(p), np.sin(p), np.cos(y), np.sin(y), np.cos(r), np.sin(r)
    return np.array([[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                     [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                     [sp, -cp*sr, cp*cr]])


def pixel_ray(u, v, width=640, height=360, fov=90.):
    if not 0 < fov < 180 or width <= 0 or height <= 0:
        raise ValueError('invalid intrinsics')
    f = width / (2 * math.tan(math.radians(fov)/2))
    return np.array([1., (u-width/2)/f, -(v-height/2)/f])


def project_ray(ray, width=640, height=360, fov=90.):
    f = width / (2 * math.tan(math.radians(fov)/2))
    if not np.isfinite(ray).all() or ray[0] <= 1e-6:
        return None
    return [width/2 + f*ray[1]/ray[0], height/2 - f*ray[2]/ray[0]]


def rotation_normalized_box(box, pose, reference_pose, width=640, height=360, fov=90.):
    """Reproject the four *box* corners to reference orientation, not 3D pose.

    A box is not an object mesh. Rotated enclosing boxes can include extra
    background; no translation/physical-depth compensation is claimed.
    """
    transform = rotation(reference_pose).T @ rotation(pose)
    pixels = [project_ray(transform @ pixel_ray(u, v, width, height, fov), width, height, fov)
              for u in (box[0], box[2]) for v in (box[1], box[3])]
    if any(p is None for p in pixels):
        return None
    return [min(p[0] for p in pixels), min(p[1] for p in pixels),
            max(p[0] for p in pixels), max(p[1] for p in pixels)]


def overlap(a, b):
    inter = max(0., min(a[2],b[2])-max(a[0],b[0])) * max(0., min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


class PairedIdentityState:
    """Frozen first-box/task + first disjoint context height references.

    The context is an observed candidate, not a verified negative identity.
    No state refresh after rejection; no previous pixel-box extrapolation.
    """
    def __init__(self):
        self.step = -1
        self.reference_pose = None
        self.task_height = None
        self.context_height = None

    def update(self, candidates, step, pose, initial_box=None, width=640, height=360, fov=90.):
        if step != self.step+1 or ((step == 0) != (initial_box is not None)):
            raise ValueError('unordered observation or future task annotation')
        rotation(pose)
        for c in candidates:
            if not set(c).issubset({'xyxy','confidence','class_id','mask','mask_pixels'}):
                raise ValueError('unexpected candidate fields (including evaluator data)')
            b = c['xyxy']
            if len(b) != 4 or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
                raise ValueError('invalid candidate box')
        self.step = step
        if step == 0:
            if len(initial_box) != 4 or not np.isfinite(initial_box).all() or initial_box[3] <= initial_box[1] or initial_box[2] <= initial_box[0]:
                raise ValueError('invalid first task box')
            self.reference_pose = dict(pose)
            self.task_height = float(initial_box[3]-initial_box[1])
            disjoint = [c for c in candidates if
                        min(c['xyxy'][2], initial_box[2]) <= max(c['xyxy'][0], initial_box[0]) or
                        min(c['xyxy'][3], initial_box[3]) <= max(c['xyxy'][1], initial_box[1])]
            if disjoint:
                c = max(disjoint, key=lambda c:c['confidence'])
                self.context_height = float(c['xyxy'][3]-c['xyxy'][1])
            selected = max(range(len(candidates)), key=lambda i:overlap(candidates[i]['xyxy'], initial_box)) if candidates else None
            if selected is not None and overlap(candidates[selected]['xyxy'], initial_box) < .5:
                selected = None
            scores, assignments = [], []
        else:
            scores = []
            for i, c in enumerate(candidates):
                normalized = rotation_normalized_box(c['xyxy'], pose, self.reference_pose, width, height, fov)
                if normalized is None:
                    continue
                h = normalized[3]-normalized[1]
                if h <= 0:
                    continue
                scores.append(dict(index=i, normalized_box=normalized, normalized_height=h,
                                   task_cost=abs(math.log(h/self.task_height)),
                                   context_cost=abs(math.log(h/self.context_height)) if self.context_height else None,
                                   confidence=float(c['confidence'])))
            assignments = []
            selected = None
            if scores:
                if self.context_height is None:
                    assignments = [dict(task_index=s['index'], context_index=None, cost=s['task_cost'],
                                        confidence_sum=s['confidence']) for s in scores]
                elif len(scores) == 1:
                    s = scores[0]
                    if s['task_cost'] <= s['context_cost']:
                        assignments = [dict(task_index=s['index'],context_index=None,cost=s['task_cost'],confidence_sum=s['confidence'])]
                else:
                    assignments = [dict(task_index=a['index'],context_index=b['index'],
                                        cost=a['task_cost']+b['context_cost'],confidence_sum=a['confidence']+b['confidence'])
                                   for a in scores for b in scores if a['index'] != b['index']]
                assignments.sort(key=lambda a:(a['cost'],-a['confidence_sum'],a['task_index'],
                                               a['context_index'] if a['context_index'] is not None else -1))
                if assignments:
                    selected = assignments[0]['task_index']
        gap = assignments[1]['cost']-assignments[0]['cost'] if len(assignments) >= 2 else None
        if selected is None:
            status = 'no_task_candidate'
        elif len(assignments) >= 2 and gap <= .10:
            status = 'assignment_conflict'
        elif step > 0 and scores and next(s['task_cost'] for s in scores if s['index'] == selected) > .30:
            status = 'shape_mismatch'
        else:
            status = 'task_candidate_available'
        return selected, dict(step=step,selected_index=selected,status=status,candidate_scores=scores,
            assignment_alternatives=assignments[:2],assignment_cost_gap=gap,task_height=self.task_height,
            context_height=self.context_height,reference_pose=self.reference_pose,gt_inputs=False,
            future_frames_available=0,task_box_source_step=0,
            scope='rotation-normalized paired shape proxy; not calibrated identity/depth')


def follow_residual(candidate, pose):
    residual = dict(x=0., y=0., z=0., yaw=0., pitch=0.)
    if candidate is None:
        residual['z'] = .08
        return residual
    b = candidate['xyxy']
    dx, dy = (b[0]+b[2])/2-320., (b[1]+b[3])/2-180.
    yaw = math.radians(pose['yaw'])
    lateral = float(np.clip(dx/320*.08, -.08, .08))
    residual.update(x=-math.sin(yaw)*lateral,y=math.cos(yaw)*lateral,
                    yaw=float(np.clip(math.degrees(math.atan2(dx,320.)), -2.,2.)),
                    pitch=float(np.clip(-math.degrees(math.atan2(dy,320.)), -1.,1.)))
    return residual


def separation_probe(candidates, indices, pose):
    """Finite action choice from a disclosed apparent-size projection proxy.

    No world target coordinates or counterfactual rendered image enters here.
    Scores describe hypothetical image separation, not observed future gains.
    """
    yaw = math.radians(pose['yaw'])
    options = [('rise',np.array([0.,0.,.08])),
               ('right',np.array([-.08*math.sin(yaw),.08*math.cos(yaw),0.])),
               ('left',np.array([.08*math.sin(yaw),-.08*math.cos(yaw),0.]))]
    r = rotation(pose)
    centers = []
    for index in indices:
        b = candidates[index]['xyxy']
        d = 320.*1.5/max(1., b[3]-b[1])
        centers.append(r @ (pixel_ray((b[0]+b[2])/2,(b[1]+b[3])/2)*d))
    if len(centers) != 2:
        return dict(x=0.,y=0.,z=.08,yaw=0.,pitch=0.), dict(chosen='rise',scores=[],reason='insufficient_two_candidate_geometry')
    scores = []
    for name, delta in options:
        p = [project_ray(r.T @ (c-delta)) for c in centers]
        if any(x is None for x in p):
            value = -1e6
        else:
            edge = min(min(x[0],640.-x[0],x[1],360.-x[1]) for x in p)
            value = float(np.linalg.norm(np.array(p[0])-p[1]))/320 - max(0.,20.-edge)/320
        scores.append(dict(action=name,projection_proxy_score=value))
    chosen = max(range(3),key=lambda i:(scores[i]['projection_proxy_score'],-i))
    delta = options[chosen][1]
    return dict(x=float(delta[0]),y=float(delta[1]),z=float(delta[2]),yaw=0.,pitch=0.), dict(
        chosen=options[chosen][0],scores=scores,reason='apparent_size_separation_proxy',
        assumed_object_height_m=1.5,is_measured_depth=False,is_information_gain=False)


def action_residual(mode, decision_index, candidates, selected_index, identity, pose, origin):
    if mode not in MODES or not 0 <= decision_index < 29:
        raise ValueError('unfrozen control or action clock')
    candidate = candidates[selected_index] if selected_index is not None else None
    zero = dict(x=0.,y=0.,z=0.,yaw=0.,pitch=0.)
    if mode == 'nominal':
        return zero, dict(branch='shared_prior',identity_state_consumed=False)
    if mode == 'fixed_rise':
        zero['z'] = .08
        return zero, dict(branch='fixed_rise',identity_state_consumed=False)
    angle = math.radians(origin['yaw'])
    if mode == 'fixed_orbit':
        zero.update(x=-.08*math.sin(angle),y=.08*math.cos(angle),yaw=-1.)
        return zero, dict(branch='fixed_orbit',identity_state_consumed=False)
    if mode == 'simple_search':
        side = -.08 if decision_index < 10 else .08 if decision_index >= 19 else 0.
        zero.update(x=-side*math.sin(angle),y=side*math.cos(angle),z=.08 if 10 <= decision_index < 19 else 0.,
                    yaw=-1. if decision_index < 10 else 1. if decision_index >= 19 else 0.)
        return zero, dict(branch='fixed_search',identity_state_consumed=False)
    if mode == 'task_visibility' or identity['status'] == 'task_candidate_available':
        return follow_residual(candidate,pose), dict(branch='task_visibility_follow',identity_state_consumed=True,
                                                    selected_candidate_consumed=selected_index)
    alternatives = identity['assignment_alternatives']
    indices = ([alternatives[0]['task_index'],alternatives[0]['context_index']]
               if alternatives and alternatives[0]['context_index'] is not None else [])
    residual, detail = separation_probe(candidates,indices,pose)
    return residual, dict(branch='identity_evidence_probe',identity_state_consumed=True,
                          identity_status_consumed=identity['status'],projection_proxy=detail)
