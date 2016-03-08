import os
import sys
import unittest

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))

from .. import matcher, outlier_detector


class TestOutlierDetector(unittest.TestCase):

    def test_distance_ratio(self):
        df = pd.DataFrame(np.array([[0, 0, 1, 1, 2, 2, 2],
                           [3, 4, 5, 6, 7, 8, 9],
                           [1.25, 10.1, 2.3, 2.4, 1.2, 5.5, 5.7]]).T,
                          columns=['source_idx', 'destination_idx', 'distance'])
        d = outlier_detector.DistanceRatio(df)
        d.compute()
        self.assertEqual(d.mask.sum(), 2)

    def test_distance_ratio_unique(self):
        data = [['A', 0, 'B', 1, 10],
                ['A', 0, 'B', 8, 10]]
        df = pd.DataFrame(data, columns=['source_image', 'source_idx',
                                         'destination_image', 'destination_idx',
                                         'distance'])
        d = outlier_detector.DistanceRatio(df)
        d.compute(0.9)
        self.assertTrue(d.mask.all() == False)

    def test_mirroring_test(self):
        # returned mask should be same length as input df
        df = pd.DataFrame(np.array([[0,0,0,1,1,1],
                           [1,2,1, 1,2,3],
                           [5,2,5,5,2,3]]).T,
                          columns=['source_idx', 'destination_idx', 'distance'])
        mask = outlier_detector.mirroring_test(df)
        self.assertEqual(mask.sum(), 1)

    def test_compute_fundamental_matrix(self):
        np.random.seed(12345)
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers,2)))
        tp = np.array(np.random.standard_normal((nbr_inliers,2)))

        F, mask = outlier_detector.compute_fundamental_matrix(fp, tp, confidence=0.5)

        np.testing.assert_array_almost_equal(F, np.array([[-0.53516611, 2.34420116, -0.60565672],
                                                          [-0.08070418, -2.77970059, 1.99678886],
                                                          [-0.89519184, 0.90058511,  1.]]))

    def tearDown(self):
        pass
