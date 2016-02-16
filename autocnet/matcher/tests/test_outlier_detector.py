import os
import sys
import unittest

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))

from .. import matcher, outlier_detector
from autocnet.examples import get_path


class TestOutlierDetector(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # actually set up everything for matches
        im1 = cv2.imread(get_path('AS15-M-0296_SML.png'))
        im2 = cv2.imread(get_path('AS15-M-0297_SML.png'))

        fd = {}

        sift = cv2.xfeatures2d.SIFT_create(10)

        fd['AS15-M-0296_SML.png'] = sift.detectAndCompute(im1, None)
        fd['AS15-M-0297_SML.png'] = sift.detectAndCompute(im2, None)

        fmatcher = matcher.FlannMatcher()
        truth_image_indices = {}
        counter = 0
        for imageid, (keypoint, descriptor) in fd.items():
            truth_image_indices[counter] = imageid
            fmatcher.add(descriptor, imageid)
            counter += 1

        fmatcher.train()
        self.matches = fmatcher.query(fd['AS15-M-0296_SML.png'][1],'AS15-M-0296_SML.png', k=3)

    def test_distance_ratio(self):
        self.assertTrue(len(outlier_detector.distance_ratio(self.matches)), 13)

    def test_distance_ratio_unique(self):
        data = [['A', 0, 'B', 1, 10],
                ['A', 0, 'B', 8, 10]]
        df = pd.DataFrame(data, columns=['source_image', 'source_idx',
                                         'destination_image', 'destination_idx',
                                         'distance'])
        mask = outlier_detector.distance_ratio(df)
        self.assertTrue(mask.all() == False)

    def test_self_neighbors(self):
        # returned mask should be same length as input df
        self.assertEquals(len(outlier_detector.self_neighbors(self.matches)), len(self.matches))

    def test_mirroring_test(self):
        # returned mask should be same length as input df
        self.assertEquals(len(outlier_detector.mirroring_test(self.matches)), len(self.matches))

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
