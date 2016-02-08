import os
import sys
import unittest
sys.path.insert(0, os.path.abspath('..'))

import numpy as np

from autocnet.examples import get_path

from .. import area


class TestArea(unittest.TestCase):

    def setUp(self):
        seed = np.random.RandomState(12345)
        self.pts = seed.rand(25,2)

    def test_area_single(self):
        total_area = 1.0
        ratio = area.convex_hull_ratio(self.pts, total_area)

        self.assertAlmostEqual(0.7566490, ratio, 5)