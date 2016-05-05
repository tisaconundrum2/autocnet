import os
import sys
import unittest

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import numpy.testing
import pandas as pd
from autocnet.transformation import transformations

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
        H = transformations.Homography(static_H,
                                       pd.DataFrame(fp, columns=['x', 'y']),
                                       pd.DataFrame(tp.T[:, :2], columns=['x', 'y']),
                                       mask=pd.Series(True, index=np.arange(fp.shape[0])))
        self.assertAlmostEqual(H.determinant, 0.6249999, 5)
        self.assertAlmostEqual(H.condition, 7.19064438, 5)

        error = H.error
        numpy.testing.assert_array_almost_equal(error['rmse'], np.zeros(20))
        self.assertAlmostEqual(error.total_rms, 0.0, 1)
        self.assertAlmostEqual(error.x_rms, 0.0, 1)
        self.assertAlmostEqual(error.y_rms, 0.0, 1)

        description = H.describe_error
        self.assertIsInstance(description, pd.DataFrame)

    def test_Homography_fail(self):
        with self.assertRaises(TypeError):
            h = transformations.Homography([1,2,3], np.arange(3), np.arange(3), None)
        with self.assertRaises(ValueError):
            h = transformations.Homography(np.arange(4).reshape(2,2),
                                           np.arange(3), np.arange(3), None)


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

        cls.F = transformations.FundamentalMatrix(static_F,
                                              pd.DataFrame(fph, columns=['x', 'y', 'h']),
                                              pd.DataFrame(tp.T, columns=['x', 'y', 'h']),
                                              mask=pd.Series(True, index=np.arange(fp.shape[0])))

    def test_f_error(self):
        self.assertIsInstance(self.F.error, pd.Series)

    def test_f_determinant(self):
        self.assertAlmostEqual(self.F.determinant, 0.624999, 5)

    def test_f_rank(self):
        # Degenerate Case
        self.assertEqual(self.F.rank, 3)

    def test_f_refine(self):
        r = self.F.refine()
        self.assertIsNone(r)
