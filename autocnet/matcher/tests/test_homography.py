import os
import numpy as np
import unittest

import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy.testing

from .. import homography


class TestHomography(unittest.TestCase):

    def setUp(self):
        pass

    def test_Homography(self):
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers,2))) #inliers

        # homography to transform fp
        static_H = np.array([[4,0.5,10],[0.25,1,5],[0.2,0.1,1]])

        #Make homogeneous
        fph = np.hstack((fp,np.ones((nbr_inliers, 1))))
        tp = static_H.dot(fph.T)
        # normalize hom. coordinates
        tp /= tp[-1,:np.newaxis]
        H = homography.Homography(static_H,
                                  fp,
                                  tp.T[:,:2])
        self.assertAlmostEqual(H.determinant, 0.6249999, 5)
        self.assertAlmostEqual(H.condition, 7.19064438, 5)
        error = H.error
        numpy.testing.assert_array_almost_equal(error['rmse'], np.zeros(20))

    def test_Homography_fail(self):
        self.assertRaises(TypeError, homography.Homography, [1,2,3], 'a', 'b')
