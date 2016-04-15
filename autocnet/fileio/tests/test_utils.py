import unittest
import pandas as pd
import numpy as np

from .. import utils

class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_slopes(self):
        x1 = pd.DataFrame({'x': np.arange(1, 11),
                           'y': np.arange(1, 11)})
        x2 = pd.DataFrame({'x': np.arange(6, 16),
                           'y': np.arange(11, 21)})

        slope = utils.calculate_slope(x1, x2)
        self.assertEqual(slope.slope[0], 2)
