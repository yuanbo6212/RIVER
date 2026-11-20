"""按冻结步表调度的刷新器，用于时钟预留闭环 A/B 两臂。

与 `river_refresh_v1.RefreshDetector` 的唯一差别：检测时刻不来自规则，
而是来自冻结的步表（A 生产偶数步 93/22，B 已验证回放 85/30）。
传输、掩码、灰度参考、身份支持状态全部沿用生产实现，不重新实现。

**不改动 `river_refresh_v1.py`**（冻结源），只在其外层包装。
"""
from __future__ import annotations

import math

import numpy as np

from river_refresh_v1 import RefreshDetector


class ScheduledRefreshDetector(RefreshDetector):
    """policy='scheduled'：`schedule` 集合内的步才真正调用检测器。

    `budget` 仍传 128 以满足父类校验，但配额逻辑被 `refresh_rule` 覆盖，
    实际调用次数由步表决定（启动时由调用方断言等于 115）。
    """

    def __init__(self, detector, schedule, budget=128):
        super().__init__(detector, "independent", budget)
        self.schedule = frozenset(int(s) for s in schedule)
        self.policy = "scheduled"

    def refresh_rule(self, step, last_detection, calls, budget, valid,
                     uncertainty, verified, motion_px, edge):
        """返回 (是否检测, 原因)。只读步表，不读身份/传输状态。"""
        if step in self.schedule:
            return True, "frozen_schedule"
        return False, "off_schedule_transport"

    def predict(self, jpeg):
        """复刻父类 predict，但调度判定改用冻结步表。

        传输、形状、审计字段与父类一致；唯一的语义差别是 refresh 的来源。
        """
        import cv2
        f = self.frame
        if f is None or jpeg != f.jpeg:
            raise ValueError("explicit original RGB frame required")
        if self.previous is not None and f.source_step <= self.previous.source_step:
            raise ValueError("causal monotonic RGB required")
        gray = cv2.cvtColor(f.rgb(), cv2.COLOR_RGB2GRAY)
        transported, valid, u = self.transport(gray)
        motion = 0.
        if self.previous is not None:
            yaw = abs((f.pose[4] - self.previous.pose[4] + 180) % 360 - 180)
            widths = [max(2., r["xyxy"][2] - r["xyxy"][0]) for r in self.observed]
            motion = math.radians(yaw) * f.intrinsic[0] / min(widths) if widths else 0.
        edge = any(r["xyxy"][0] < 8 or r["xyxy"][2] > f.width - 8 for r in transported)
        refresh, reason = self.refresh_rule(f.source_step, self.last_detection,
                                            self.calls, self.budget, valid, u,
                                            self.verified, motion, edge)
        if refresh:
            shape, observed = self.detector.predict(jpeg)
            self.calls += 1
            self.last_detection = f.source_step
            kind = "detector"
        else:
            shape = (f.height, f.width)
            observed = transported if valid else []
            kind = "RGB_LK" if valid else "empty_unsupported"
        self.audit.append(dict(
            step=f.source_step, policy=self.policy, budget=self.budget,
            detector_called=refresh, reason=reason, proposal_origin=kind,
            transport_valid=valid, transport_uncertainty=u,
            observed_camera_motion_box_fraction=motion, verified_before_frame=self.verified,
            proposal_count=len(observed), cumulative_detector_calls=self.calls,
            source_frame_id=f.frame_id,
            parent_observation_frame_id=self.previous.frame_id if self.previous is not None else None,
            parent_observation_jpeg_sha256=self.previous.jpeg_sha256 if self.previous is not None else None,
            source_jpeg_sha256=f.jpeg_sha256, capture_ns=f.capture_mono_ns,
            GT_inputs=False, identity_is_recomputed_on_current_RGB=True,
            schedule_source="frozen_step_list"))
        self.observed = observed
        self.gray = gray
        self.previous = f
        return shape, observed
