"""Causal detector refresh with shared current-RGB LK proposal transport.

No actor/GT/map inputs. Transported masks are explicitly distinguished from
fresh detector masks; task identity is recomputed by the unchanged HSV reader.
"""
import math,time
import numpy as np

POLICIES=('full','periodic','independent','coupled')

def refresh_rule(policy,step,last_detection,calls,budget,valid,uncertainty,verified,motion_px,edge):
    if policy not in POLICIES or budget not in (64,128,230):raise ValueError('fixed policy/budget required')
    if step==0:return True,'bootstrap'
    quota=min(budget,2+math.floor(step*(budget-2)/229))
    if calls>=quota:return False,'paced_budget'
    if policy=='full':return True,'full_frequency'
    interval=math.ceil(230/budget)
    if policy=='periodic':return step%interval==0,'fixed_period'
    age=step-last_detection
    if not valid:return True,'transport_failure'
    if age>=4:return True,'maximum_detector_age'
    if uncertainty>.15:return True,'transport_uncertainty'
    if policy=='independent':return (not verified),'identity_support'
    if motion_px>.15 or edge:return True,'view_motion_exposure'
    # Unknown identity also warrants an affordable actual detector refresh.
    return (not verified),'identity_support'

class RefreshDetector:
    def __init__(self,detector,policy,budget):
        self.detector=detector;self.policy=policy;self.budget=budget
        self.frame=None;self.previous=None;self.gray=None;self.observed=[]
        self.calls=0;self.last_detection=-1;self.verified=False
        self.audit=[];self.pause_step=None

    def update_identity(self,readout):
        self.verified=bool(readout.get('supported') and readout.get('selected_index') is not None)

    def transport(self,gray):
        import cv2
        if self.gray is None or not self.observed:return [],False,1.
        h,w=gray.shape;out=[];errors=[]
        for row in self.observed:
            mask=np.asarray(row['mask'],np.uint8)
            points=cv2.goodFeaturesToTrack(self.gray,maxCorners=24,qualityLevel=.02,minDistance=3,mask=mask)
            if points is None or len(points)<6:return [],False,1.
            q,s,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,points,None,winSize=(15,15),maxLevel=2)
            if q is None:return [],False,1.
            back,s2,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,q,None,winSize=(15,15),maxLevel=2)
            if back is None:return [],False,1.
            ok=(s.ravel()!=0)&(s2.ravel()!=0)&(np.linalg.norm(back-points,axis=2).ravel()<=1.)
            if ok.sum()<6 or ok.mean()<.6:return [],False,1.
            delta=(q-points).reshape(-1,2)[ok];shift=np.median(delta,axis=0)
            x1,y1,x2,y2=row['xyxy'];scale=max(2.,min(x2-x1,y2-y1))
            errors.append(float(np.quantile(np.linalg.norm(delta-shift,axis=1),.9)/scale))
            moved=cv2.warpAffine(mask,np.array([[1,0,shift[0]],[0,1,shift[1]]],np.float32),(w,h),flags=cv2.INTER_NEAREST)
            yy,xx=np.nonzero(moved)
            if len(xx)<6:return [],False,1.
            out.append(dict(row,xyxy=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1],mask=moved))
        return out,True,max(errors)

    def predict(self,jpeg):
        import cv2
        f=self.frame
        if f is None or jpeg!=f.jpeg:raise ValueError('explicit original RGB frame required')
        if self.previous is not None and f.source_step<=self.previous.source_step:raise ValueError('causal monotonic RGB required')
        started=time.monotonic_ns();gray=cv2.cvtColor(f.rgb(),cv2.COLOR_RGB2GRAY)
        transported,valid,u=self.transport(gray)
        motion=0.
        if self.previous is not None:
            yaw=abs((f.pose[4]-self.previous.pose[4]+180)%360-180)
            widths=[max(2.,r['xyxy'][2]-r['xyxy'][0]) for r in self.observed]
            motion=math.radians(yaw)*f.intrinsic[0]/min(widths) if widths else 0.
        edge=any(r['xyxy'][0]<8 or r['xyxy'][2]>f.width-8 for r in transported)
        refresh,reason=refresh_rule(self.policy,f.source_step,self.last_detection,self.calls,self.budget,valid,u,self.verified,motion,edge)
        if refresh:
            shape,observed=self.detector.predict(jpeg)
            self.calls+=1;self.last_detection=f.source_step;kind='detector'
        else:
            shape=(f.height,f.width);observed=transported if valid else [];kind='RGB_LK' if valid else 'empty_unsupported'
        self.pause_step=f.source_step if self.policy=='coupled' and self.verified and not valid and not refresh else None
        self.audit.append(dict(step=f.source_step,policy=self.policy,budget=self.budget,detector_called=refresh,
            reason=reason,proposal_origin=kind,transport_valid=valid,transport_uncertainty=u,
            observed_camera_motion_box_fraction=motion,verified_before_frame=self.verified,
            proposal_count=len(observed),cumulative_detector_calls=self.calls,source_frame_id=f.frame_id,
            parent_observation_frame_id=self.previous.frame_id if self.previous is not None else None,
            parent_observation_jpeg_sha256=self.previous.jpeg_sha256 if self.previous is not None else None,
            source_jpeg_sha256=f.jpeg_sha256,capture_ns=f.capture_mono_ns,finished_ns=time.monotonic_ns(),
            GT_inputs=False,identity_is_recomputed_on_current_RGB=True))
        self.observed=observed;self.gray=gray;self.previous=f
        return shape,observed
