"""Shared readout correctness guard; no model, simulator or GT dependencies."""
import math


def early_transport_skip(source, current, now, max_age_s=2., max_steps=20, max_current_age_s=.15):
    if type(now) is not int or now < source.capture_mono_ns or now < current.capture_mono_ns:
        return 'invalid_or_future_clock'
    if current.source_step < source.source_step:
        return 'nonmonotone_source'
    if (now-source.capture_mono_ns)/1e9 > max_age_s or current.source_step-source.source_step > max_steps:
        return 'transport_horizon'
    if (now-current.capture_mono_ns)/1e9 > max_current_age_s:
        return 'current_image_stale'
    return None


def protect_current_cheap(cheap, probabilities, source_frame_id, current_frame_id):
    """Never veto same-capture deep; unknown old support cannot veto fresh acceptance."""
    p=list(probabilities)
    if not p or any(not math.isfinite(x) or x<0 for x in p) or not math.isclose(sum(p),1.,abs_tol=1e-9):
        raise ValueError('normalized finite null-plus-candidate distribution required')
    return (source_frame_id != current_frame_id and cheap.get('supported') is True
        and type(cheap.get('selected_index')) is int and cheap['selected_index']>=0
        and max(range(len(p)),key=lambda i:p[i])==0)
