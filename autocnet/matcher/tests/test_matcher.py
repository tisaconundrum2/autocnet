import os
import sys
import unittest

import cv2
import numpy as np

sys.path.append(os.path.abspath('..'))

from .. import matcher
from autocnet.examples import get_path


class TestMatcher(unittest.TestCase):

    def setUp(self):
        im1 = cv2.imread(get_path('AS15-M-0296_SML.png'))
        im2 = cv2.imread(get_path('AS15-M-0297_SML.png'))

        self.fd = {}

        sift = cv2.xfeatures2d.SIFT_create(10)

        self.fd['AS15-M-0296_SML.png'] = sift.detectAndCompute(im1, None)
        self.fd['AS15-M-0297_SML.png'] = sift.detectAndCompute(im2, None)

    def test_flann_match_k_eq_2(self):
        fmatcher = matcher.FlannMatcher()
        source_image = self.fd['AS15-M-0296_SML.png']
        fmatcher.add(source_image[1], 0)

        self.assertTrue(len(fmatcher.nid_lookup), 1)

        fmatcher.train()

        self.assertRaises(ValueError, fmatcher.query,self.fd['AS15-M-0296_SML.png'][1],0, k=2 )
        matches = fmatcher.query(self.fd['AS15-M-0297_SML.png'][1], 1, k=2)
        self.assertEqual(len(matches), 20)

    def tearDown(self):
        pass


