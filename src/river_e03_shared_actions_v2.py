"""Readback-robust bounded-horizon shared camera actuator, recovery v2.

Derived from frozen river_e03_shared_actions.py; parent SHA256:
6d5e6239c2348b50523be35852eac9f34e05fe978a0234db37a58a27dd2b7eec
Immutable intents/limits/mailbox are aliases of the parent, preserving callers.
No physical limits or readback validation tolerances are enlarged.
"""
from __future__ import annotations
import math
import numpy as np
from river_e03_shared_actions import (ActionLimits, CurrentPointing, ActionIntent,
    CommandMailbox, FIELDS, DIRECTIONS, _require, _pose, _angle, _sensor_frame,
    _brake_distance, _speed_for_distance, _slew_velocity)

PARENT_SOURCE_SHA256 = "6d5e6239c2348b50523be35852eac9f34e05fe978a0234db37a58a27dd2b7eec"
DEFAULT_MAX_ACTION_TICKS = 461


def _robust_parameters(rate, accel, dt, error):
    eta = error / dt
    command_rate = rate - eta
    command_slew = accel * dt - eta
    brake_drop = accel * dt - 2. * eta
    _require(command_rate > 0. and brake_drop > 0.,
             "readback bound too large for robust acceleration reserve")
    return eta, command_rate, command_slew, brake_drop


def _robust_angular_interval(distance_negative, distance_positive, rate, accel, dt, error):
    eta, command_rate, slew, drop = _robust_parameters(rate, accel, dt, error)
    # B(s) includes the present actual tick.  Worst actual rate is u+eta;
    # future worst actual outward rate decreases by drop, until nonpositive.
    upper = min(command_rate, _speed_for_distance(distance_positive, rate, drop/dt, dt) - eta)
    lower = max(-command_rate, -_speed_for_distance(distance_negative, rate, drop/dt, dt) + eta)
    return lower, upper, slew


class SharedActionExecutor:
    """Actual pose generator plus post-effect measured-path accountant.

    Default unknown identity brakes; an explicit, valid predeclared_search
    intent permits fixed lateral motion without identity, never pointing.
    Stale/future/invalid evidence does not authorize that exception. No old
    pointing target is retained. Common prior is zero/not called.
    """
    def __init__(self,origin,*,limits=None,mailbox=None,pose_decoder=None,max_action_ticks=DEFAULT_MAX_ACTION_TICKS):
        self.origin=_pose(origin);self.limits=limits or ActionLimits()
        _require(type(max_action_ticks) is int and max_action_ticks > 0,"finite positive actuator horizon required")
        self.max_action_ticks=max_action_ticks
        _require(max_action_ticks*self.limits.readback_position_tolerance_m < self.limits.total_path_m,
                 "readback horizon exhausts total path budget before motion")
        _robust_parameters(self.limits.speed_m_s,self.limits.accel_m_s2,self.limits.tick_s,
                           self.limits.readback_position_tolerance_m)
        for axis in ("yaw","pitch"):
            _robust_parameters(getattr(self.limits,axis+"_rate_deg_s"),getattr(self.limits,axis+"_accel_deg_s2"),
                               self.limits.tick_s,self.limits.readback_angle_tolerance_deg)
        self.mailbox=mailbox or CommandMailbox();self.pose_decoder=pose_decoder
        yaw=math.radians(self.origin["yaw"])
        self.right=np.array([-math.sin(yaw),math.cos(yaw),0.])
        self.velocity=0.;self.angular_velocity=dict(yaw=0.,pitch=0.)
        self.path_m=0.;self.command_path_m=0.;self.audit=[];self._pending=None
        self._active=None;self._last_post=None;self._ids=set();self.failed=False

    def tick_start(self,*,current_frame_id,sim_s,mono_ns,measured_pose):
        _require(not self.failed and self._pending is None,"failed executor or missing post_tick readback")
        _require(len(self.audit)<self.max_action_ticks,"finite actuator horizon exhausted; no further command authorized")
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
        ticks_left=self.max_action_ticks-len(self.audit)
        # Triangle inequality: actual length <= command length + position error.
        # Reserve one full accepted error for every remaining effect, including
        # zero commands. This is a finite execution horizon, not infinite hold.
        path_noise_reserve=ticks_left*limits.readback_position_tolerance_m
        command_budget=remaining-path_noise_reserve
        _require(command_budget>=-1e-10,"insufficient finite-horizon readback path reserve")
        eta,command_rate,command_slew,brake_drop=_robust_parameters(
            limits.speed_m_s,limits.accel_m_s2,dt,limits.readback_position_tolerance_m)
        cap=_speed_for_distance(max(0.,command_budget),command_rate,brake_drop/dt,dt)
        new_velocity=_slew_velocity(self.velocity,desired_direction*command_rate,command_slew/dt,dt,-cap,cap)
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
            lower,upper,slew=_robust_angular_interval(extent+offset,extent-offset,rate,accel,dt,
                limits.readback_angle_tolerance_deg)
            velocity=_slew_velocity(self.angular_velocity[axis],desired,slew/dt,dt,lower,upper)
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
            kinematic_camera_only=True,gt_inputs=False,
            actuator_version="readback_robust_v2",parent_source_sha256=PARENT_SOURCE_SHA256,
            max_action_ticks=self.max_action_ticks,remaining_action_ticks=ticks_left,
            path_readback_reserve_m=path_noise_reserve,command_path_budget_m=max(0.,command_budget),
            robustness_scope="bounded_per_effect_readback_error_and_finite_action_horizon")
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
