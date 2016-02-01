import os
import sys
import unittest

sys.path.append(os.path.abspath('..'))

import numpy as np

from .. import subpixel as sp


class TestSubPixel(unittest.TestCase):

    def setup(self):
        pass

    def test_clip_roi(self):
        img = np.arange(10000).reshape(100,100)
        center = (30,30)

        clip = sp.clip_roi(img, center, 9)
        self.assertEqual(clip.mean(), 2979.5)

        center = (55.4, 63.1)
        clip = sp.clip_roi(img, center, 27)
        self.assertEqual(clip.mean(), 5512.5)

        self.assertRaises(ValueError, sp.clip_roi, img, center, 10)