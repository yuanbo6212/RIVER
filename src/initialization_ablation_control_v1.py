"""Three initialization policies with one shared later evidence-update rule.

Control for the October BalancedExitControl / CalibratedRoute chain only.
It is not integrated with November SharedIdentityReadout / feedback_v2.
All modes delegate to the original reference_non_goal implementation. The
only scientific intervention is direction assignment immediately before step 0.
"""
import math

MODES=('task_initialization','fixed_left_initialization','fixed_right_initialization')

def initial_direction(mode,box,width,height):
    if mode not in MODES:
        raise ValueError('Unknown frozen initialization mode')
    if box is None or len(box)!=4 or not all(math.isfinite(float(x)) for x in box):
        raise ValueError('Finite first-frame task rectangle required')
    x0,y0,x1,y1=map(float,box)
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):
        raise ValueError('First-frame task rectangle outside image')
    if mode=='task_initialization':
        return -1. if (x0+x1)/2>=width/2 else 1.
    return -1. if mode=='fixed_left_initialization' else 1.

class InitializationAblationControl:
    def __init__(self,url,token,scale,mode,origin,sensor_attributes,infer=None):
        if mode not in MODES:
            raise ValueError('Unknown frozen initialization mode')
        from development_extension_control_20261017 import DevelopmentExtensionControl
        self.core=DevelopmentExtensionControl(url,token,scale,'reference_non_goal',origin,sensor_attributes,infer)
        self.mode=mode
        self.width=int(sensor_attributes['image_size_x'])
        self.height=int(sensor_attributes['image_size_y'])
        self.initialized=False
        self.next_step=0
        self.d0=None

    @property
    def path_length(self):
        return self.core.path_length

    def observe(self,jpeg,step,pose,initial_box,candidates,selected,identity,neural_reply):
        if step!=self.next_step:
            raise ValueError('Noncausal observation order')
        if step==0:
            self.d0=initial_direction(self.mode,initial_box,self.width,self.height)
            # Same calibrated route object used by the original adaptive arm.
            self.core.core.impl.direction=self.d0
            self.initialized=True
        elif not self.initialized or initial_box is not None:
            raise ValueError('Initialization can consume only the first task box')
        command,trace=self.core.observe(jpeg,step,pose,initial_box,candidates,selected,identity,neural_reply)
        self.next_step+=1
        trace=dict(trace,initialization_ablation=dict(mode=self.mode,input_step=0,d0=self.d0,
            later_direction_updates='original_reference_non_goal',
            direction_overwritten_after_step0=False))
        return command,trace
