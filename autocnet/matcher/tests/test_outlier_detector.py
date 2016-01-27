import os
import sys
import unittest

import cv2

sys.path.append(os.path.abspath('..'))

from .. import matcher, outlier_detector
from autocnet.examples import get_path


class TestOutlierDetector(unittest.TestCase):

    def setUp(self):
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

    def test_self_neighbors(self):
        print(self.matches[outlier_detector.self_neighbors(self.matches)])
        # returned mask should be same length as input df
        self.assertEquals(len(outlier_detector.self_neighbors(self.matches)), len(self.matches))

    def test_mirroring_test(self):
        # returned mask should be same length as input df
        self.assertEquals(len(outlier_detector.mirroring_test(self.matches)), len(self.matches))

    def tearDown(self):
        pass