import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from river_e03_metadecision_v1 import choose

class Tests(unittest.TestCase):
    def test_joint_can_skip_unknown_when_compute_is_harmful(self):
        risks=dict(adaptive_left=.2,adaptive_right=.4,scene_feedback=.6)
        a=choose('J3-independent',risks,target_verified=False,compute_feasible=True)
        b=choose('J4-task-value',risks,target_verified=False,compute_feasible=True)
        self.assertTrue(a['compute']);self.assertEqual(b['direction'],'Left')
    def test_joint_can_compute_when_scene_changes_best_downstream_policy(self):
        risks=dict(adaptive_left=.7,adaptive_right=.5,scene_feedback=.1)
        self.assertTrue(choose('J4-task-value',risks,target_verified=True,compute_feasible=True)['compute'])
        self.assertFalse(choose('J3-independent',risks,target_verified=True,compute_feasible=True)['compute'])
    def test_shared_budget_and_tie_fallback(self):
        risks=dict(adaptive_left=.3,adaptive_right=.2,scene_feedback=.0)
        for p in ('J3-independent','J4-task-value','J3-full-scene'):
            r=choose(p,risks,target_verified=False,compute_feasible=False)
            self.assertEqual(r['direction'],'Right');self.assertFalse(r['compute'])
        self.assertFalse(choose('J4-task-value',dict(adaptive_left=.2,adaptive_right=.3,scene_feedback=.2),target_verified=False,compute_feasible=True)['compute'])

if __name__=='__main__':unittest.main()
