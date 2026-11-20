import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from river_refresh_v1 import refresh_rule

class Tests(unittest.TestCase):
    def test_current_rgb_transport_and_provenance(self):
        import io,json
        import numpy as np
        from PIL import Image
        from river_refresh_v1 import RefreshDetector
        from river_e03_perception import FramePacket
        try:import cv2
        except ImportError:self.skipTest('image transport test runs in remote OpenCV environment')
        rng=np.random.RandomState(3);image=rng.randint(0,255,(96,160,3),dtype=np.uint8)
        mask=np.zeros((96,160),np.uint8);mask[20:65,30:75]=1
        class Detector:
            calls=0
            def predict(self,jpeg):
                self.calls+=1;return (96,160),[dict(xyxy=[30,20,75,65],mask=mask,confidence=.9,class_id=2)]
        def frame(step,rgb):
            f=io.BytesIO();Image.fromarray(rgb).save(f,format='JPEG',quality=100)
            return FramePacket(str(step),step,step*.1,step*100000000,f.getvalue(),160,96,(0,0,3,0,0,0),(80,0,80,0,80,48,0,0,1))
        d=Detector();p=RefreshDetector(d,'periodic',64);p.frame=frame(0,image);p.predict(p.frame.jpeg)
        p.frame=frame(1,np.roll(image,3,axis=1));shape,rows=p.predict(p.frame.jpeg)
        self.assertEqual(d.calls,1);self.assertEqual(len(rows),1)
        self.assertAlmostEqual(rows[0]['xyxy'][0],33,delta=1)
        self.assertEqual(p.audit[-1]['proposal_origin'],'RGB_LK');self.assertEqual(p.audit[-1]['source_frame_id'],'1')
        self.assertFalse(p.audit[-1]['GT_inputs'])
    def test_bootstrap_and_budget(self):
        for policy in ('full','periodic','independent','coupled'):
            self.assertTrue(refresh_rule(policy,0,-1,0,64,False,1,False,0,False)[0])
            self.assertFalse(refresh_rule(policy,229,228,64,64,False,1,False,0,True)[0])
    def test_motion_changes_coupled_decision(self):
        args=(10,9,1,64,True,.01,True,.3,False)
        self.assertFalse(refresh_rule('independent',*args)[0]);self.assertTrue(refresh_rule('coupled',*args)[0])
    def test_periodic_does_not_use_identity(self):
        self.assertTrue(refresh_rule('periodic',8,4,2,64,True,0,True,0,False)[0])
        self.assertFalse(refresh_rule('periodic',9,8,2,64,False,1,False,0,True)[0])
    def test_pacing_bounds_over_whole_task(self):
        for policy in ('periodic','independent','coupled'):
            for budget in (64,128):
                calls=0;last=-1
                for step in range(230):
                    go,_=refresh_rule(policy,step,last,calls,budget,False,1,False,1,True)
                    if go:calls+=1;last=step
                self.assertLessEqual(calls,budget)

if __name__=='__main__':unittest.main()
