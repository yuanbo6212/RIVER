"""Motion gate v2: a fresh timestamp alone never authorises a hold.

The v1 rule was

    lateral = 'Stay' if feedback and fresh else controller.direction

where `fresh` only checked the pointing OBJECT and its timestamps.
`CurrentPointing.from_light()` returns a timestamped object even when no
candidate was selected, in which case `world_bearing is None`. A fresh empty
pointing was therefore treated as a valid hold basis, and the controller stayed
put while the executor downstream recorded `identity_unknown`.

v2 separates four cases explicitly:

  usable        pointing exists, world_bearing is not None, and the reference
                is fresh  -> hold, and point at the bearing
  stale         pointing exists but is outside the age bound -> not usable
  empty         pointing exists and is fresh but world_bearing is None
                -> NOT a hold basis; fall through to the declared search
  brake         near the terminal brake time -> Stay, no pointing

`empty` must never be authorised as `identity_conditioned`. This module does not
change any threshold, budget, refresh policy or executor safety check; it only
decides which of the four cases applies.
"""
from __future__ import annotations

import math

MAX_REFERENCE_AGE_S = 0.2


def _fresh(pointing, mono_ns, sim_s):
    if pointing is None:
        return False
    obs_mono = getattr(pointing, "observation_mono_ns", None)
    obs_sim = getattr(pointing, "observation_sim_s", None)
    if obs_mono is None or obs_sim is None:
        return False
    if not (0.0 <= (mono_ns - obs_mono) / 1e9 <= MAX_REFERENCE_AGE_S):
        return False
    if not (0.0 <= sim_s - obs_sim <= MAX_REFERENCE_AGE_S + 1e-9):
        return False
    return mono_ns <= pointing.expires_mono_ns


def pointing_is_usable(pointing, mono_ns, sim_s):
    """A pointing is usable only if it is fresh AND carries a real bearing."""
    if not _fresh(pointing, mono_ns, sim_s):
        return False
    bearing = getattr(pointing, "world_bearing", None)
    if bearing is None:
        return False
    if len(bearing) != 3:
        return False
    return all(math.isfinite(float(v)) for v in bearing)


def classify_state(pointing, mono_ns, sim_s, phase):
    """Return one of: brake, usable_hold, empty_fresh, stale, absent."""
    if phase != "search":
        return "brake"
    if pointing is None:
        return "absent"
    if not _fresh(pointing, mono_ns, sim_s):
        return "stale"
    if getattr(pointing, "world_bearing", None) is None:
        return "empty_fresh"
    return "usable_hold"


def control_spec_v2(controller, issue, feedback):
    """Drop-in replacement for river_e03_feedback_v1.control_spec.

    Same signature and same return keys, so callers need no other change.
    """
    first = controller.bootstrap[0] if controller.bootstrap else None
    elapsed = issue["sim_s"] - first.capture_sim_s if first else 0.0
    phase = "terminal_brake" if elapsed >= 22.4 - 1e-9 else controller.motion_phase

    p = controller.pointing
    state = classify_state(p, issue["mono_ns"], issue["sim_s"], phase)

    if state == "brake":
        return dict(phase=phase, lateral="Stay", pointing=None, state=phase,
                    motion_gate="terminal_brake")

    if not feedback:
        # the fixed-search arms keep their original semantics exactly
        usable = state == "usable_hold"
        return dict(phase=phase,
                    lateral=controller.direction,
                    pointing=p if usable else None,
                    state="search",
                    motion_gate=state)

    if state == "usable_hold":
        return dict(phase=phase, lateral="Stay", pointing=p,
                    state="identity_hold", motion_gate="usable_hold")

    # empty_fresh, stale and absent all fall through to the declared search.
    # An empty pointing is never labelled identity_hold and never authorises
    # identity_conditioned motion.
    return dict(phase=phase, lateral=controller.direction, pointing=None,
                state="search", motion_gate=state)


def spec_to_intent_kwargs(spec):
    """The motion basis the caller must attach to the issued intent.

    Only a usable hold may be identity-conditioned; everything else is the
    pre-declared search, which the executor already validates against its
    speed/acceleration/path/braking limits.
    """
    return dict(motion_basis="identity_conditioned"
                if spec["state"] == "identity_hold" else "predeclared_search")
