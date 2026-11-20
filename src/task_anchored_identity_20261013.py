"""Task-bound first-frame appearance and explicit non-target comparisons.

Development repair/ablation, not claimed novel or calibrated. Reference RGB
and the task rectangle are sufficient even when the detector misses the task.
No pose prediction, hidden identity labels, future frames or template updates.
"""
import numpy as np
from cosfly_pi05.causal_reentry_identity import rgb_histogram_bins,masked_histogram

MODES=('task_anchored_hsv','contrastive_rank_hsv','contrastive_abstain_hsv')

def box_iou(a,b):
    area=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-area
    return area/union if union>0 else 0.

class TaskAnchoredIdentity:
    def __init__(self,mode='task_anchored_hsv'):
        if mode not in MODES:raise ValueError('Unregistered development readout')
        self.mode=mode;self.step=-1;self.template=None;self.negatives=[];self.reference_source=None

    def update(self,rgb,candidates,step,pose,initial_box=None,fov=90.):
        if step!=self.step+1 or ((step==0)!=(initial_box is not None)):
            raise ValueError('Unordered frame or later task annotation')
        image=np.asarray(rgb)
        if image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected RGB')
        for c in candidates:
            if set(c)-{'xyxy','confidence','class_id','mask'}:raise ValueError('Candidate contains non-observation fields')
            b=np.asarray(c['xyxy'],float)
            if b.shape!=(4,) or not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1] or c['mask'].shape!=image.shape[:2]:
                raise ValueError('Invalid observed box or mask')
        bins=rgb_histogram_bins(image) if candidates or step==0 else None
        descriptors={i:masked_histogram(bins,c['mask']) for i,c in enumerate(candidates) if int(c['mask'].sum())>=8}
        initial_selection=None
        if step==0:
            b=np.asarray(initial_box,float);h,w=image.shape[:2]
            if b.shape!=(4,) or not np.isfinite(b).all() or not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):
                raise ValueError('Invalid task rectangle')
            roi=np.zeros((h,w),bool)
            roi[int(np.floor(b[1])):int(np.ceil(b[3])),int(np.floor(b[0])):int(np.ceil(b[2]))]=True
            if int(roi.sum())<8:raise ValueError('Task reference has too few pixels')
            supported=[]
            for i,c in enumerate(candidates):
                overlap=box_iou(c['xyxy'],b);inside=np.asarray(c['mask'],bool)&roi
                if i in descriptors and overlap>0. and int(inside.sum())>=8:
                    supported.append((overlap,c['confidence'],-i,i,inside))
                elif i in descriptors and overlap==0. and not inside.any():
                    # Spatially outside the initial designated rectangle;
                    # this is not an oracle actor-ID assignment.
                    self.negatives.append(descriptors[i].copy())
            if supported:
                winner=max(supported,key=lambda t:t[:3]);initial_selection=winner[3]
                self.template=masked_histogram(bins,winner[4]);self.reference_source='task_intersected_initial_candidate'
            else:
                self.template=masked_histogram(bins,roi);self.reference_source='initial_task_rectangle_RGB_fallback'
        self.step=step
        scores=[]
        for i,hist in descriptors.items():
            pos=float(np.sqrt(hist*self.template).sum())
            neg=max((float(np.sqrt(hist*r).sum()) for r in self.negatives),default=0.)
            evidence=pos-neg
            scores.append(dict(index=i,positive_similarity=pos,known_negative_similarity=neg,
                positive_minus_negative=evidence,selection_score=pos if self.mode=='task_anchored_hsv' else evidence))
        if step==0:
            selected=initial_selection
            reason='task_supported_initial_selection' if selected is not None else 'reference_retained_detector_missed_task'
            margin=None
        elif not scores:
            selected=None;reason='no_valid_current_candidate';margin=None
        else:
            ordered=sorted(scores,key=lambda r:(r['selection_score'],candidates[r['index']]['confidence'],-r['index']),reverse=True)
            best=ordered[0];selected=best['index']
            margin=best['selection_score']-ordered[1]['selection_score'] if len(ordered)>1 else None
            rejected=self.mode=='contrastive_abstain_hsv' and (best['positive_minus_negative']<=0. or (margin is not None and margin<=0.))
            if rejected:selected=None;reason='no_strict_positive_advantage_or_tied_candidates'
            else:reason=self.mode
        return selected,dict(step=step,mode=self.mode,reason=reason,selected_index=selected,
            template_registered=True,reference_source=self.reference_source,reference_box_source_step=0,
            negative_reference_count=len(self.negatives),template_updated=False,negative_memory_updated=False,
            candidate_scores=scores,top_margin=margin,calibrated_identity_probability=False,
            gt_inputs=False,future_frames_available=0,negative_labels_from_initial_task_disjointness_only=True)
