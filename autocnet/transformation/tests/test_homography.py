import os
import sys
import unittest
import warnings
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd

import autocnet.transformation.homography as hm

class TestHomography(unittest.TestCase):

    def test_Homography(self):
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers, 2)))  # inliers

        # homography to transform fp
        static_H = np.array([[4, 0.5, 10], [0.25, 1, 5], [0.2, 0.1, 1]])

        # Make homogeneous
        fph = np.hstack((fp, np.ones((nbr_inliers, 1))))
        tp = static_H.dot(fph.T)
        # normalize hom. coordinates
        tp /= tp[-1, :np.newaxis]
        H, mask = hm.compute_homography(fph, tp.T)
        np.testing.assert_array_almost_equal(H, static_H)

        error = hm.compute_error(H, fph, tp.T)
        np.testing.assert_array_almost_equal(error['rmse'], np.zeros(20))
        self.assertAlmostEqual(error.total_rms, 0.0, 1)
        self.assertAlmostEqual(error.x_rms, 0.0, 1)
        self.assertAlmostEqual(error.y_rms, 0.0, 1)
