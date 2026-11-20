"""Development re-entry identity association from current RGB proposals.

Only an initial/previously accepted visual template and current proposal
masks are used. CARLA labels never enter this module. The fixed thresholds
are development hypotheses, not validated test-set settings.
"""

import numpy as np

from .geometry import project_from_carla_pose


def rgb_histogram_bins(rgb):
    """Vectorized OpenCV-compatible coarse HSV bins without a cv2 dependency."""
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image")
    image = image / 255.
    high = np.max(image, axis=2)
    low = np.min(image, axis=2)
    delta = high - low
    hue = np.zeros_like(high)
    nonzero = delta > 1e-8
    red = nonzero & (high == image[:, :, 0])
    green = nonzero & (high == image[:, :, 1]) & ~red
    blue = nonzero & ~red & ~green
    hue[red] = np.mod((image[:, :, 1][red] - image[:, :, 2][red]) / delta[red], 6.)
    hue[green] = (image[:, :, 2][green] - image[:, :, 0][green]) / delta[green] + 2.
    hue[blue] = (image[:, :, 0][blue] - image[:, :, 1][blue]) / delta[blue] + 4.
    hue = hue / 6.
    saturation = np.divide(delta, high, out=np.zeros_like(high), where=high > 1e-8)
    h = np.minimum(9, (hue * 10).astype(np.int32))
    s = np.minimum(3, (saturation * 4).astype(np.int32))
    v = np.minimum(15, (high * 16).astype(np.int32))
    return (h * 64 + s * 16 + v).astype(np.int32)


def masked_histogram(bins, mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != bins.shape or int(mask.sum()) < 8:
        raise ValueError("Template/proposal mask needs at least eight pixels")
    counts = np.bincount(bins[mask], minlength=640).astype(np.float64)
    return counts / counts.sum()


def _unique_proposals(masks, scores, threshold=.85):
    coarse = [np.asarray(mask, dtype=bool)[::4, ::4] for mask in masks]
    unique = []
    for index in sorted(range(len(masks)), key=lambda i: (-scores[i], i)):
        mask = coarse[index]
        duplicate = False
        for other in unique:
            intersection = np.count_nonzero(mask & coarse[other])
            union = np.count_nonzero(mask | coarse[other])
            if union and intersection / union >= threshold:
                duplicate = True
                break
        if duplicate:
            continue
        unique.append(index)
    return unique


class CausalReentryIdentity:
    """Conservative single-frame association; no hidden identity oracle."""

    def __init__(self, appearance_threshold=.8, ambiguity_margin=.03,
                 min_pixels=8, initial_uncertainty_m=1.,
                 uncertainty_growth_m_per_decision=.6,
                 motion_gate_enabled=True):
        if (not 0 <= appearance_threshold <= 1 or ambiguity_margin < 0 or
                min_pixels < 8 or initial_uncertainty_m < 0 or
                uncertainty_growth_m_per_decision < 0):
            raise ValueError("Invalid development re-entry settings")
        self.appearance_threshold = float(appearance_threshold)
        self.ambiguity_margin = float(ambiguity_margin)
        self.min_pixels = int(min_pixels)
        self.initial_uncertainty_m = float(initial_uncertainty_m)
        self.uncertainty_growth_m_per_decision = float(
            uncertainty_growth_m_per_decision)
        self.motion_gate_enabled = bool(motion_gate_enabled)
        self.references = []

    def motion_compatible(self, candidate_world_xyz, predicted_world_xyz,
                          missing_age):
        """Causal age-growing uncertainty gate applied after visual ranking."""
        candidate = np.asarray(candidate_world_xyz, dtype=np.float64)
        predicted = np.asarray(predicted_world_xyz, dtype=np.float64)
        if (candidate.shape != (3,) or predicted.shape != (3,) or
                not np.isfinite(candidate).all() or
                not np.isfinite(predicted).all() or missing_age < 1):
            raise ValueError("Invalid causal motion evidence")
        distance = float(np.linalg.norm(candidate - predicted))
        radius = (self.initial_uncertainty_m +
                  self.uncertainty_growth_m_per_decision * missing_age)
        return distance <= radius, {"distance_to_causal_prediction_m": distance,
                                    "allowed_uncertainty_radius_m": radius,
                                    "missing_age": int(missing_age)}

    def add_verified_visual(self, rgb, mask):
        """Call only for frame-zero or causally accepted masks, never GT masks."""
        hist = masked_histogram(rgb_histogram_bins(rgb), mask)
        if len(self.references) < 2:
            self.references.append(hist)
        else:
            self.references[1] = hist

    def propose_points(self, predicted_world_xyz, pose, intrinsic, shape):
        height, width = shape
        center, depth = project_from_carla_pose(
            np.asarray(predicted_world_xyz, dtype=np.float64).reshape(1, 3),
            pose, intrinsic)
        if not np.isfinite(center).all() or depth[0] <= .2:
            return []
        x, y = center[0]
        return [[float(x + dx), float(y + dy)]
                for dy in (-20, 0, 20) for dx in (-80, -40, 0, 40, 80, 120)
                if 0 <= x + dx < width and 0 <= y + dy < height]

    def choose(self, rgb, proposals):
        """Return (accepted mask or None, diagnostics) from GT-free proposals."""
        if not self.references:
            return None, {"reason": "no_causal_reference"}
        bins = rgb_histogram_bins(rgb)
        valid = []
        for index, item in enumerate(proposals):
            mask = np.asarray(item["mask"], dtype=bool)
            if mask.shape != bins.shape or int(mask.sum()) < self.min_pixels:
                continue
            hist = masked_histogram(bins, mask)
            similarity = max(float(np.sqrt(hist * ref).sum())
                             for ref in self.references)
            valid.append((index, mask, similarity,
                          float(item["sam2_score"])))
        if not valid:
            return None, {"reason": "no_valid_proposal"}
        keep = _unique_proposals([x[1] for x in valid], [x[3] for x in valid])
        order = sorted(keep, key=lambda i: (-valid[i][2], valid[i][0]))
        best = valid[order[0]]
        runner_up = valid[order[1]][2] if len(order) > 1 else 0.
        margin = best[2] - runner_up
        accepted = (best[2] >= self.appearance_threshold and
                    margin >= self.ambiguity_margin)
        return (best[1] if accepted else None), {
            "reason": "appearance_reentry" if accepted else "ambiguous_or_low_score",
            "proposal_count": len(proposals), "unique_count": len(order),
            "selected_index": best[0] if accepted else None,
            "best_appearance": best[2], "margin": margin,
            "threshold": self.appearance_threshold,
            "required_margin": self.ambiguity_margin}
