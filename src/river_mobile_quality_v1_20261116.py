"""One shared moving controller; three frozen detector refresh policies."""
from river_clock_schedule_v1 import ScheduledRefreshDetector
from river_refresh_v1 import refresh_rule

ARMS = ('full230', 'independent128', 'uniform128')
UNIFORM_STEPS = tuple(i * 229 // 127 for i in range(128))
assert len(set(UNIFORM_STEPS)) == 128 and UNIFORM_STEPS[0] == 0 and UNIFORM_STEPS[-1] == 229


class MobileQualityDetector(ScheduledRefreshDetector):
    def __init__(self, detector, arm):
        if arm not in ARMS:
            raise ValueError('frozen three-arm comparison only')
        super().__init__(detector, UNIFORM_STEPS, budget=230 if arm == 'full230' else 128)
        self.arm = arm
        self.policy = arm

    def transport(self, gray):
        if self.arm == 'full230':
            return [], False, 0.
        return super().transport(gray)

    def refresh_rule(self, step, last_detection, calls, budget, valid,
                     uncertainty, verified, motion_px, edge):
        if self.arm == 'full230':
            return True, 'full_frequency_without_unused_LK'
        if self.arm == 'uniform128':
            return step in self.schedule, 'uniform_frozen_step' if step in self.schedule else 'uniform_transport'
        return refresh_rule('independent', step, last_detection, calls, 128,
                            valid, uncertainty, verified, motion_px, edge)

    def predict(self, jpeg):
        result = super().predict(jpeg)
        if self.calls > (230 if self.arm == 'full230' else 128):
            raise RuntimeError('actual detector budget exceeded')
        self.audit[-1].update(comparison_arm=self.arm, physical_translation_allowed=True)
        return result
