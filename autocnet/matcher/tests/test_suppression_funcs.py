import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))
from .. import suppression_funcs as sf


class Test_Suppression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        r = np.random.RandomState(12345)
        cls.df = pd.DataFrame(r.uniform(0, 1, (10, 4)), columns=['response',
                                                                 'correlation',
                                                                 'distance',
                                                                 'error'])

    def test_response(self):
        col = self.df.apply(sf.response, axis=1, args=[None])
        self.assertEqual(col.all(), self.df['response'].all())

    def test_correlation(self):
        col = self.df.apply(sf.correlation, axis=1, args=[None])
        self.assertEqual(col.all(), self.df['correlation'].all())

    def test_distance(self):
        col = self.df.apply(sf.distance, axis=1, args=[None])
        self.assertEqual(col.all(), self.df['distance'].all())

    def test_error(self):
        col = self.df.apply(sf.error, axis=1, args=[None])
        self.assertEqual(col.all(), self.df['error'].all())

