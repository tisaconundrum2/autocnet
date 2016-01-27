import os
import sys
import unittest

import cv2

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
        truth_image_indices = {}
        counter = 0
        for imageid, (keypoint, descriptor) in self.fd.items():
            truth_image_indices[counter] = imageid
            fmatcher.add(descriptor, imageid)
            counter += 1

        self.assertEqual(truth_image_indices, fmatcher.image_indices)

        fmatcher.train()

        matched = fmatcher.query(self.fd['AS15-M-0296_SML.png'][1],'AS15-M-0296_SML.png', k=2)
        matched_to = matched['source_image']
        self.assertTrue(matched_to[matched_to == 'AS15-M-0296_SML.png'].any())
        self.assertEqual(7, len(matched))

        # Check that self neighbors are being omitted
        distance = matched['distance']
        self.assertFalse(distance[distance == 0].any())

    def test_flann_match_k_eq_3(self):
        fmatcher = matcher.FlannMatcher()
        truth_image_indices = {}
        counter = 0
        for imageid, (keypoint, descriptor) in self.fd.items():
            truth_image_indices[counter] = imageid
            fmatcher.add(descriptor, imageid)
            counter += 1

        self.assertEqual(truth_image_indices, fmatcher.image_indices)

        fmatcher.train()

        matched = fmatcher.query(self.fd['AS15-M-0296_SML.png'][1], 'AS15-M-0296_SML.png', k=3)
        self.assertEqual(12, len(matched))

        # Check that self neighbors are being omitted
        distance = matched['distance']
        self.assertFalse(distance[distance == 0].any())

    def tearDown(self):
        pass

class TestOutlierDetector(unittest.TestCase):

    def setUp(self):
        #actually set up everything for matches
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

        self.outliers = matcher.OutlierDetector()

    def test_distance_ratio(self):
        self.assertTrue(len(self.outliers.distance_ratio(self.matches)), 13)

    def test_self_neighbors(self):
        print(self.matches[self.outliers.self_neighbors(self.matches)])
        #returned mask should be same length as input df
        self.assertEquals(len(self.outliers.self_neighbors(self.matches)), len(self.matches))


    def test_mirroring_test(self):
        #returned mask should be same length as input df
        self.assertEquals(len(self.outliers.mirroring_test(self.matches)), len(self.matches))


    def tearDown(self):
        pass
