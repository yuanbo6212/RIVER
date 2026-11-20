"""Shared search/hold controller for all adaptive arms; no evaluator inputs."""

def control_spec(controller,issue,feedback):
    first=controller.bootstrap[0] if controller.bootstrap else None
    elapsed=issue['sim_s']-first.capture_sim_s if first else 0.
    phase='terminal_brake' if elapsed>=22.4-1e-9 else controller.motion_phase
    p=controller.pointing;now=issue['mono_ns']
    fresh=p is not None and now<=p.expires_mono_ns and 0<=(now-p.observation_mono_ns)/1e9<=.2 and 0<=issue['sim_s']-p.observation_sim_s<=.2+1e-9
    if phase!='search':return dict(phase=phase,lateral='Stay',pointing=None,state=phase)
    lateral='Stay' if feedback and fresh else controller.direction
    return dict(phase=phase,lateral=lateral,pointing=p if fresh else None,state='identity_hold' if feedback and fresh else 'search')
