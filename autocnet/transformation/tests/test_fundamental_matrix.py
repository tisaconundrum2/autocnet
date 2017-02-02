import os
import sys
import unittest
import warnings
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd

import autocnet.transformation.fundamental_matrix as fm

class TestFundamentalMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers, 2)))  # inliers

        static_F = np.array([[4, 0.5, 10], [0.25, 1, 5], [0.2, 0.1, 1]])

        # Make homogeneous
        fph = np.hstack((fp, np.ones((nbr_inliers, 1))))
        tp = static_F.dot(fph.T)
        # normalize hom. coordinates
        tp /= tp[-1, :np.newaxis]

        cls.F = np.array([[-0.685892, -5.870193, 2.268333],
                          [-0.704199, 12.88776,  -3.040341],
                          [-0.231815, -2.806056, 1.]])

        cls.x1 = pd.DataFrame(fph, columns=['x', 'y', 'h'])
        cls.x2 = pd.DataFrame(tp.T, columns=['x', 'y', 'h'])

    def test_compute_f(self):
        np.random.seed(12345)
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers, 2)))
        tp = np.array(np.random.standard_normal((nbr_inliers, 2)))

        F, mask = fm.compute_fundamental_matrix(fp, tp, method='ransac')

        np.testing.assert_array_almost_equal(F, self.F)

    def test_compute_mle_f(self):
        #TODO: Write a better test for MLE the data here is too clean.
        np.random.seed(12345)
        nbr_inliers = 20
        fp = pd.DataFrame(np.array(np.random.standard_normal((nbr_inliers, 2))))
        tp = pd.DataFrame(np.array(np.random.standard_normal((nbr_inliers, 2))))

        F, mask = fm.compute_fundamental_matrix(fp, tp, method='mle')

        np.testing.assert_array_almost_equal(F, self.F)

    def test_f_error(self):
        #TODO: This is a stochastic process - how to test?
        pass
