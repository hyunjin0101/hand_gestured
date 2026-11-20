"""Unit tests for deterministic 198-feature extraction and overlay clipping."""
import unittest
from types import SimpleNamespace
import numpy as np
from gesture_ar.features import extract_features
from gesture_ar.common import overlay


class TestFeatures(unittest.TestCase):
    def test_dimensions_and_finiteness(self):
        landmarks=[SimpleNamespace(x=0.1+i*0.01,y=0.15+i*0.02,z=i*0.002,visibility=0)
                   for i in range(21)]
        result=extract_features(SimpleNamespace(landmark=landmarks))
        self.assertEqual(result.shape,(198,))
        self.assertTrue(np.isfinite(result).all())

    def test_identical_joints_do_not_nan(self):
        points=[SimpleNamespace(x=.5,y=.5,z=0,visibility=0) for i in range(21)]
        result=extract_features(SimpleNamespace(landmark=points))
        self.assertTrue(np.isfinite(result).all())

    def test_sprite_clipped_at_negative_offset(self):
        frame=np.zeros((6,6,3),dtype=np.uint8)
        sprite=np.full((4,4,4),255,dtype=np.uint8)
        overlay(frame,sprite,-2,-2)
        self.assertEqual(frame[0,0].tolist(),[255,255,255])
        self.assertEqual(frame[-1,-1].tolist(),[0,0,0])


if __name__=='__main__': unittest.main()
