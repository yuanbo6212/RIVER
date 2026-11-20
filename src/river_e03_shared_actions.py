"""Shared causal L/Stay/R camera actions; no model, simulator or prior RPC.

This is a constrained kinematic camera adapter, not a flight-dynamics model.
Only the world owner calls tick_start/post_tick. Producers submit immutable
intents through a one-slot nonblocking mailbox. Ground truth is not an input.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import threading

import numpy as np

from river_geometry import camera_frame_from_carla_pose

FIELDS=("x","y","z","pitch","yaw","roll")
DIRECTIONS={"Left":-1.,"Stay":0.,"Right":1.}


def _require(test,message):
    if not test:raise ValueError(message)


def _pose(value):
    _require(hasattr(value,"keys") and set(value)==set(FIELDS),"six camera telemetry fields only")
    result={key:float(value[key]) for key in FIELDS}
    _require(all(math.isfinite(v) for v in result.values()),"nonfinite camera telemetry")
    _require(abs(result["roll"])<=.002,"zero-roll camera mount required")
    return result


def _angle(value):return (value+180.)%360.-180.


def _sensor_frame(value):
    # Live owner IDs are decimal; archived technical receipts may prefix
    # case_NNN/sensor_NNN. Both preserve the actual sensor frame ordinal.
    tail=str(value).rsplit('_',1)[-1]
    _require(tail.isdecimal(),"actual numeric sensor frame binding required")
    return int(tail)


@dataclass(frozen=True)
class ActionLimits:
    tick_s:float=.05
    total_path_m:float=3.
    speed_m_s:float=.8
    accel_m_s2:float=1.6
    yaw_rate_deg_s:float=20.
    pitch_rate_deg_s:float=10.
    yaw_accel_deg_s2:float=40.
    pitch_accel_deg_s2:float=20.
    yaw_offset_deg:float=15.
    pitch_offset_deg:float=10.
    max_reference_age_s:float=.2
    readback_position_tolerance_m:float=.0001
    readback_angle_tolerance_deg:float=.002

    def __post_init__(self):
        _require(all(math.isfinite(v) and v>0 for v in self.__dict__.values()),"positive finite limits required")
        _require(self.total_path_m<=3.,"cannot enlarge shared three-metre budget")


@dataclass(frozen=True)
class CurrentPointing:
    """An age-bounded observed bearing, not a current target position.

    A future causal transport may supply the same structure with its validated
    latest-observed epoch and provenance. `current_frame_id` means that
    observation epoch, NEVER an unobserved current owner world frame.
    """
    current_frame_id:str
    source_frame_id:str
    capture_sim_s:float
    capture_mono_ns:int
    delivered_mono_ns:int
    expires_mono_ns:int
    world_bearing:tuple[float,float,float] | None
    evidence_ids:tuple[str,...]
    source_kind:str="same_frame_light"
    observation_sim_s:float | None=None
    observation_mono_ns:int | None=None

    def __post_init__(self):
        _require(self.source_kind in {"same_frame_light","validated_causal_transport"},"unregistered pointing source")
        _require(bool(self.current_frame_id) and bool(self.source_frame_id),"frame binding required")
        _require(all(type(v) is int for v in (self.capture_mono_ns,self.delivered_mono_ns,self.expires_mono_ns)) and
                 0<=self.capture_mono_ns<=self.delivered_mono_ns<=self.expires_mono_ns,
                 "invalid evidence capture/delivery/deadline clock")
        _require(math.isfinite(self.capture_sim_s) and self.capture_sim_s>=0,"invalid source simulation time")
        _require(self.source_kind!="same_frame_light" or self.current_frame_id==self.source_frame_id,"old light bbox is not current")
        if self.source_kind=="same_frame_light":
            _require(self.observation_sim_s in (None,self.capture_sim_s) and self.observation_mono_ns in (None,self.capture_mono_ns),
                     "light observation epoch cannot be re-stamped")
            object.__setattr__(self,"observation_sim_s",self.capture_sim_s)
            object.__setattr__(self,"observation_mono_ns",self.capture_mono_ns)
        else:
            _require(self.observation_sim_s is not None and math.isfinite(self.observation_sim_s) and self.observation_sim_s>=self.capture_sim_s,
                     "transport requires its real latest observed epoch")
            _require(type(self.observation_mono_ns) is int and self.capture_mono_ns<=self.observation_mono_ns<=self.delivered_mono_ns,
                     "transport observation acquisition time required")
        _require(_sensor_frame(self.source_frame_id)<=_sensor_frame(self.current_frame_id),"transport observation precedes source")
        if self.world_bearing is not None:
            ray=np.asarray(self.world_bearing,float)
            _require(ray.shape==(3,) and np.isfinite(ray).all() and abs(np.linalg.norm(ray)-1.)<=1e-8,"unit observed bearing required")
            _require(bool(self.evidence_ids),"accepted pointing needs source evidence IDs")

    @classmethod
    def from_light(cls,snapshot,*,delivered_mono_ns,expires_mono_ns):
        from river_e03_perception import LightSnapshot
        _require(isinstance(snapshot,LightSnapshot),"original immutable LightSnapshot required")
        detail=json.loads(snapshot.identity_json)
        future=detail.get("future_frames_available")
        _require(detail.get("gt_inputs") is False and
                 (future is False or (type(future) is int and future==0)),"noncausal identity receipt")
        _require(not set(detail)&{"task","depth_swap","exit_mode","target_px","distractor_px",
            "target_box","distractor_box","target_pose","actor_id"},"evaluator-only identity fields prohibited")
        selected=detail.get("selected_index")
        ray=None
        if selected is not None:
            _require(type(selected) is int and 0<=selected<len(snapshot.candidates),"bad current selected index")
            box=snapshot.candidates[selected].xyxy
            pixel=np.array([(box[0]+box[2])/2.,(box[1]+box[3])/2.,1.])
            optical=np.linalg.solve(np.asarray(snapshot.frame.intrinsic).reshape(3,3),pixel)
            pose=dict(zip(FIELDS,snapshot.frame.pose))
            camera=camera_frame_from_carla_pose(pose,timestamp=snapshot.frame.capture_sim_s,world_frame_id="carla")
            world=camera.world_from_optical@optical;ray=tuple(world/np.linalg.norm(world))
        frame=snapshot.frame
        return cls(frame.frame_id,frame.frame_id,frame.capture_sim_s,frame.capture_mono_ns,
            delivered_mono_ns,expires_mono_ns,ray,(frame.frame_id+":"+snapshot.candidate_sha256,))


@dataclass(frozen=True)
class ActionIntent:
    command_id:str
    lateral:str
    issued_mono_ns:int
    issued_sim_s:float
    expires_mono_ns:int
    pointing:CurrentPointing | None
    motion_basis:str="identity_conditioned"

    def __post_init__(self):
        _require(isinstance(self.command_id,str) and bool(self.command_id),"stable command ID required")
        _require(self.lateral in DIRECTIONS,"only Left/Stay/Right allowed")
        _require(self.motion_basis in {"identity_conditioned","predeclared_search"},"unregistered motion basis")
        _require(type(self.issued_mono_ns) is int and 0<=self.issued_mono_ns<=self.expires_mono_ns,"invalid intent deadline")
        _require(type(self.expires_mono_ns) is int and (self.pointing is None or isinstance(self.pointing,CurrentPointing)),"invalid action receipt")
        _require(math.isfinite(self.issued_sim_s) and self.issued_sim_s>=0,"invalid issue simulation clock")


class CommandMailbox:
    """One pending intent, latest replaces earlier. Neither side waits."""
    def __init__(self):
        self._lock=threading.Lock();self._pending=None

    def submit(self,intent):
        _require(isinstance(intent,ActionIntent),"immutable ActionIntent required")
        if not self._lock.acquire(False):return dict(admitted=False,reason="mailbox_busy")
        try:
            replaced=self._pending.command_id if self._pending is not None else None
            self._pending=intent
            return dict(admitted=True,replaced_command_id=replaced)
        finally:self._lock.release()

    def poll(self):
        if not self._lock.acquire(False):return None
        try:
            result=self._pending;self._pending=None;return result
        finally:self._lock.release()


def _brake_distance(speed,accel,dt):
    """Current tick plus all future minimum-deceleration tick displacements."""
    count=int(math.ceil(max(0.,speed)/(accel*dt)))
    return dt*sum(max(0.,speed-j*accel*dt) for j in range(count))


def _speed_for_distance(distance,cap,accel,dt):
    lo,hi=0.,cap
    for _ in range(45):
        middle=(lo+hi)/2.
        if _brake_distance(middle,accel,dt)<=max(0.,distance):lo=middle
        else:hi=middle
    return lo


def _slew_velocity(previous,desired,accel,dt,lower,upper):
    limited=max(previous-accel*dt,min(previous+accel*dt,desired))
    limited=max(lower,min(upper,limited))
    _require(abs(limited-previous)<=accel*dt+1e-8,"insufficient reserved braking space")
    return limited


class SharedActionExecutor:
    """Actual pose generator plus post-effect measured-path accountant.

    Default unknown identity brakes; an explicit, valid predeclared_search
    intent permits fixed lateral motion without identity, never pointing.
    Stale/future/invalid evidence does not authorize that exception. No old
    pointing target is retained. Common prior is zero/not called.
    """
    def __init__(self,origin,*,limits=None,mailbox=None,pose_decoder=None):
        self.origin=_pose(origin);self.limits=limits or ActionLimits()
        self.mailbox=mailbox or CommandMailbox();self.pose_decoder=pose_decoder
        yaw=math.radians(self.origin["yaw"])
        self.right=np.array([-math.sin(yaw),math.cos(yaw),0.])
        self.velocity=0.;self.angular_velocity=dict(yaw=0.,pitch=0.)
        self.path_m=0.;self.command_path_m=0.;self.audit=[];self._pending=None
        self._active=None;self._last_post=None;self._ids=set();self.failed=False

    def tick_start(self,*,current_frame_id,sim_s,mono_ns,measured_pose):
        _require(not self.failed and self._pending is None,"failed executor or missing post_tick readback")
        pose=_pose(measured_pose);dt=self.limits.tick_s;limits=self.limits
        _require(isinstance(current_frame_id,str) and bool(current_frame_id),"current owner frame required")
        _require(type(mono_ns) is int and mono_ns>=0 and math.isfinite(sim_s) and sim_s>=0,"owner physical clocks required")
        if self._last_post is not None:
            _require(current_frame_id==self._last_post["frame_id"] and abs(sim_s-self._last_post["sim_s"])<1e-8,
                     "owner start must continue previous effect snapshot")
            _require(mono_ns>=self._last_post["mono_ns"],"owner monotonic clock moved backwards")
        incoming=self.mailbox.poll();rejection=None
        if incoming is not None:
            if incoming.command_id in self._ids:rejection="duplicate_command_id"
            elif incoming.issued_mono_ns>mono_ns or incoming.issued_sim_s>sim_s+1e-9:rejection="future_command"
            else:self._ids.add(incoming.command_id);self._active=incoming
        intent=self._active;fallback=rejection;pointing=None
        if rejection is not None:intent=None
        elif intent is None:fallback="no_intent"
        elif mono_ns>intent.expires_mono_ns:fallback="intent_expired"
        elif intent.pointing is None:fallback="identity_unknown"
        else:
            evidence=intent.pointing
            if evidence.capture_mono_ns>mono_ns or evidence.delivered_mono_ns>mono_ns or evidence.capture_sim_s>sim_s+1e-9:
                fallback="future_evidence";self._active=None
            elif mono_ns>evidence.expires_mono_ns:fallback="evidence_expired"
            elif (_sensor_frame(evidence.current_frame_id)>_sensor_frame(current_frame_id) or
                  evidence.observation_sim_s>sim_s+1e-9 or evidence.observation_mono_ns>mono_ns):
                fallback="future_reference_frame";self._active=None
            elif (sim_s-evidence.observation_sim_s>limits.max_reference_age_s+1e-9 or
                  (mono_ns-evidence.observation_mono_ns)/1e9>limits.max_reference_age_s+1e-9):fallback="evidence_age_exceeded"
            elif evidence.world_bearing is None:fallback="identity_unknown"
            else:pointing=evidence
        search_authorized=(intent is not None and intent.motion_basis=="predeclared_search" and
                           fallback in (None,"identity_unknown"))
        motion_authorized=pointing is not None or search_authorized
        desired_direction=DIRECTIONS[intent.lateral] if motion_authorized else 0.
        remaining=max(0.,limits.total_path_m-self.path_m)
        cap=_speed_for_distance(remaining,limits.speed_m_s,limits.accel_m_s2,dt)
        new_velocity=_slew_velocity(self.velocity,desired_direction*limits.speed_m_s,limits.accel_m_s2,dt,-cap,cap)
        requested=dict(pose);delta=self.right*new_velocity*dt
        for key,value in zip(("x","y","z"),delta):requested[key]+=float(value)
        target_angles=None
        if pointing is not None:
            x,y,z=pointing.world_bearing
            target_angles=dict(yaw=math.degrees(math.atan2(y,x)),pitch=math.degrees(math.atan2(z,math.hypot(x,y))))
        new_angular={}
        for axis in ("yaw","pitch"):
            offset=_angle(pose[axis]-self.origin[axis]);extent=getattr(limits,axis+"_offset_deg")
            _require(abs(offset)<=extent+limits.readback_angle_tolerance_deg,"camera outside shared angular envelope")
            rate=getattr(limits,axis+"_rate_deg_s");accel=getattr(limits,axis+"_accel_deg_s2")
            desired=0. if target_angles is None else max(-rate,min(rate,_angle(target_angles[axis]-pose[axis])/dt))
            upper=_speed_for_distance(extent-offset,rate,accel,dt)
            lower=-_speed_for_distance(extent+offset,rate,accel,dt)
            velocity=_slew_velocity(self.angular_velocity[axis],desired,accel,dt,lower,upper)
            requested[axis]=pose[axis]+velocity*dt;new_angular[axis]=velocity
        requested["roll"]=0.
        # Optional deployment decoder converts this already constrained pose,
        # never computes a hidden prior action. It cannot alter its six fields.
        if self.pose_decoder is not None:
            decoded=_pose(self.pose_decoder(dict(requested)))
            _require(all(abs(decoded[k]-requested[k])<1e-10 for k in FIELDS),"pose decoder changed shared constrained command")
            requested=decoded
        row=dict(current_frame_id=current_frame_id,issue_sim_s=sim_s,issue_mono_ns=mono_ns,
            expected_effect_sim_s=sim_s+dt,
            command_id=intent.command_id if intent is not None else None,
            intent_issued_mono_ns=intent.issued_mono_ns if intent is not None else None,
            intent_expires_mono_ns=intent.expires_mono_ns if intent is not None else None,
            evidence_source_frame_id=pointing.source_frame_id if pointing is not None else None,
            evidence_current_frame_id=pointing.current_frame_id if pointing is not None else None,
            evidence_current_means="latest_observed_epoch_not_owner_world_frame",
            evidence_ids=list(pointing.evidence_ids) if pointing is not None else [],
            evidence_age_sim_s=sim_s-pointing.capture_sim_s if pointing is not None else None,
            reference_age_sim_s=sim_s-pointing.observation_sim_s if pointing is not None else None,
            reference_age_wall_s=(mono_ns-pointing.observation_mono_ns)/1e9 if pointing is not None else None,
            source_kind=pointing.source_kind if pointing is not None else None,
            requested_pose=dict(requested),start_measured_pose=pose,requested_velocity_m_s=new_velocity,
            requested_angular_velocity_deg_s=new_angular,fallback_reason=fallback,
            intended_lateral=intent.lateral if intent is not None else "Stay",path_m_before=self.path_m,
            motion_basis=intent.motion_basis if intent is not None else "identity_conditioned",
            motion_authorized=motion_authorized,predeclared_search_authorized=search_authorized,
            search_does_not_assert_target_position=search_authorized,
            requested_step_m=float(np.linalg.norm(delta)),remaining_path_m=remaining,
            prior_model_called=False,prior_executed=False,prior_value="common_zero",
            kinematic_camera_only=True,gt_inputs=False)
        received=intent.pointing if intent is not None else None
        row['received_reference']=None if received is None else dict(
            source_frame_id=received.source_frame_id,observation_frame_id=received.current_frame_id,
            capture_sim_s=received.capture_sim_s,capture_mono_ns=received.capture_mono_ns,
            observation_sim_s=received.observation_sim_s,observation_mono_ns=received.observation_mono_ns,
            delivered_mono_ns=received.delivered_mono_ns,expires_mono_ns=received.expires_mono_ns,
            source_age_sim_s=sim_s-received.capture_sim_s,reference_age_sim_s=sim_s-received.observation_sim_s,
            reference_age_wall_s=(mono_ns-received.observation_mono_ns)/1e9,
            evidence_ids=list(received.evidence_ids),source_kind=received.source_kind,
            consumed=pointing is not None)
        self._pending=row
        return dict(requested),dict(row)

    def post_tick(self,*,frame_id,sim_s,mono_ns,measured_pose):
        _require(self._pending is not None and not self.failed,"no outstanding executed tick")
        row=self._pending;pose=_pose(measured_pose);limits=self.limits;dt=limits.tick_s
        _require(frame_id!=row["current_frame_id"] and abs(sim_s-row["issue_sim_s"]-dt)<1e-7 and mono_ns>=row["issue_mono_ns"],
                 "effect must be next physical tick without deadline extension")
        start=row["start_measured_pose"];delta=np.array([pose[k]-start[k] for k in ("x","y","z")])
        distance=float(np.linalg.norm(delta));self.path_m+=distance;self.command_path_m+=row["requested_step_m"]
        actual_velocity=float(delta@self.right/dt)
        angular={axis:_angle(pose[axis]-start[axis])/dt for axis in ("yaw","pitch")}
        position_error=float(np.linalg.norm([pose[k]-row["requested_pose"][k] for k in ("x","y","z")]))
        angle_error=max(abs(_angle(pose[k]-row["requested_pose"][k])) for k in ("pitch","yaw","roll"))
        failures=[]
        if self.path_m>limits.total_path_m+1e-9:failures.append("actual_path_budget_exceeded")
        if position_error>limits.readback_position_tolerance_m+1e-12:failures.append("position_readback_mismatch")
        if angle_error>limits.readback_angle_tolerance_deg+1e-12:failures.append("angle_readback_mismatch")
        if distance/dt>limits.speed_m_s+2*limits.readback_position_tolerance_m/dt:failures.append("actual_speed_exceeded")
        if abs(actual_velocity-self.velocity)/dt>limits.accel_m_s2+2*limits.readback_position_tolerance_m/dt**2:failures.append("actual_acceleration_exceeded")
        for axis in angular:
            if abs(angular[axis])>getattr(limits,axis+"_rate_deg_s")+2*limits.readback_angle_tolerance_deg/dt:failures.append(axis+"_rate_exceeded")
            if abs(angular[axis]-self.angular_velocity[axis])/dt>getattr(limits,axis+"_accel_deg_s2")+2*limits.readback_angle_tolerance_deg/dt**2:failures.append(axis+"_acceleration_exceeded")
        row.update(effect_frame_id=frame_id,effect_sim_s=sim_s,effect_mono_ns=mono_ns,
            readback_pose=pose,actual_step_m=distance,actual_path_m=self.path_m,command_path_m=self.command_path_m,
            actual_velocity_m_s=actual_velocity,actual_angular_velocity_deg_s=angular,
            position_readback_error_m=position_error,angle_readback_error_deg=angle_error,violations=failures)
        self.audit.append(row);self._pending=None;self.velocity=actual_velocity;self.angular_velocity=angular
        self._last_post=dict(frame_id=frame_id,sim_s=sim_s,mono_ns=mono_ns);self.failed=bool(failures)
        if failures:raise RuntimeError("action_readback_invalid: "+",".join(failures))
        return dict(row)
