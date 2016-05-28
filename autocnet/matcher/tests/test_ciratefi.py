import unittest
import warnings
import math

import cv2
import numpy as np

from scipy.ndimage.interpolation import rotate

from .. import ciratefi
from autocnet.examples import get_path
from autocnet.matcher import subpixel as sp
from scipy.ndimage.interpolation import rotate
from scipy.misc import imresize
from scipy.misc import imread

class TestCiratefi(unittest.TestCase):
    '''
    ciratefi(s_template, d_search, upsampling=10., alpha=math.pi/8,
                     cifi_thresh=80, rafi_thresh=50, tefi_thresh=100,
                     use_percentile=True, radii=list(range(1,3)))
    '''

    @classmethod
    def setUpClass(cls):
        im1 = imread(get_path('AS15-M-0298_SML.png'), flatten=True)
        im2 = imread(get_path('AS15-M-0297_SML.png'), flatten=True)

        im1_coord = (482.09783936, 652.40679932)
        im2_coord = (262.19442749, 652.44750977)

        cls.template = sp.clip_roi(im1, im1_coord, 17)
        cls.template = rotate(cls.template, 90)
        cls.template = imresize(cls.template, 1.)

        cls.search = sp.clip_roi(im1, im1_coord, 21)
        cls.search = rotate(cls.search, -90)
        cls.search = imresize(cls.search, 1.)

        cls.upsampling = 10
        cls.alpha = math.pi/8
        cls.cifi_thresh = 98
        cls.rafi_thresh = 90
        cls.tefi_thresh = 100
        cls.use_percentile = True

        cls.cifi_number_of_warnings = 2
        cls.rafi_number_of_warnings = 2

    def test_cifi(self):
            # check all warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ciratefi.cifi(self.template, self.search, .99, radii=[100], use_percentile=False)
                self.assertEqual(len(w), self.cifi_number_of_warnings)

            # Threshold out of bounds error
            self.assertRaises(ValueError, ciratefi.cifi, self.template, self.search, -1.1, use_percentile=False)
            # radii list empty/none error
            self.assertRaises(ValueError, ciratefi.cifi, self.template, self.search, 90, radii=None)
            # scales list empty/none error
            self.assertRaises(ValueError, ciratefi.cifi, self.template, self.search, 90, scales=None)
            # template is bigger than search error
            self.assertRaises(ValueError, ciratefi.cifi, self.search, self.template, -1.1, use_percentile=False)

            with warnings.catch_warnings(record=True) as w:
                pixels, scales = ciratefi.cifi(self.template, self.search, thresh=self.cifi_thresh,
                                               radii=range(1, 8), use_percentile=True)

                warnings.simplefilter("always")
                self.assertEqual(len(w), 0)

                self.assertEqual(self.search.shape, scales.shape)
                self.assertIn((np.floor(self.search.shape[0]/2), np.floor(self.search.shape[1]/2)), pixels)
                self.assertTrue(pixels.size in range(0, self.search.size))

    def test_rafi(self):
        rafi_pixels = [(5, 6), (5, 16), (10, 10), (11, 16), (16, 6), (16, 7),
                       (16, 8), (16, 13), (16, 14)]

        rafi_scales = np.ones(self.search.shape, dtype=float)

        # check all warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ciratefi.rafi(self.template, self.search, rafi_pixels,
                          rafi_scales, thresh=1, radii=[100],
                          use_percentile=False)
            self.assertEqual(len(w), self.rafi_number_of_warnings)

        # Threshold out of bounds error
        self.assertRaises(ValueError, ciratefi.rafi, self.template, self.search, rafi_pixels, rafi_scales,
                          -1.1, use_percentile=False)

        # Radii list is empty.None error
        self.assertRaises(ValueError, ciratefi.rafi, self.search, self.template, rafi_pixels, -1.1, radii=None)
        # candidate pixel list empty/none error
        self.assertRaises(ValueError, ciratefi.rafi, self.template, self.search, None, rafi_scales)
        # scales list empty/none error
        self.assertRaises(ValueError, ciratefi.rafi, self.template, self.search, rafi_pixels, None)
        # template is bigger than search error
        self.assertRaises(ValueError, ciratefi.rafi, self.search, self.template, rafi_pixels, rafi_scales)
        # best scale nd array is not equal image shape
        self.assertRaises(ValueError, ciratefi.rafi, self.template, self.search, rafi_pixels, rafi_scales[:10])

        with warnings.catch_warnings(record=True) as w:
            pixels, scales = ciratefi.rafi(self.template, self.search, rafi_pixels, rafi_scales,
                                           thresh=self.rafi_thresh, radii=range(1, 8), use_percentile=True,
                                           alpha=self.alpha)

            warnings.simplefilter("always")
            self.assertEqual(len(w), 0)

            self.assertIn((np.floor(self.search.shape[0]/2), np.floor(self.search.shape[1]/2)), pixels)
            self.assertTrue(pixels.size in range(0, self.search.size))

    def test_tefi(self):
        tefi_pixels = [(10, 10)]
        tefi_scales = np.ones(self.search.shape, dtype=float)
        tefi_angles = [3.14159265]

        # Threshold out of bounds error
        self.assertRaises(ValueError, ciratefi.tefi, self.template, self.search, tefi_pixels, tefi_scales,
                          tefi_angles, thresh=-1.1, use_percentile=False, alpha=self.alpha)

        # angle list is empty/None error
        self.assertRaises(ValueError, ciratefi.tefi, self.search, self.template, tefi_pixels, tefi_scales, None)
        # candidate pixel list empty/none error
        self.assertRaises(ValueError, ciratefi.tefi, self.template, self.search, None, tefi_scales, tefi_angles)
        # scales list empty/none error
        self.assertRaises(ValueError, ciratefi.tefi, self.template, self.search, tefi_pixels, None, tefi_angles)
        # template is bigger than search error
        self.assertRaises(ValueError, ciratefi.tefi, self.search, self.template, tefi_pixels, tefi_scales, -1.1)
        # best scale nd array is smaller than number of candidate pixels
        self.assertRaises(ValueError, ciratefi.tefi, self.template, self.search, tefi_pixels, tefi_scales[:10], -1.1)

        with warnings.catch_warnings(record=True) as w:
            pixel = ciratefi.tefi(self.template, self.search, tefi_pixels, tefi_scales, tefi_angles,
                                           thresh=self.tefi_thresh, use_percentile=True, alpha=self.alpha,
                                           upsampling=self.upsampling)

            warnings.simplefilter("always")
            self.assertEqual(len(w), 5)

            self.assertIn((np.floor(self.search.shape[0]/2), np.floor(self.search.shape[1]/2)), pixel)
            self.assertTrue(pixel[0][0] == 10 and pixel[0][1] == 10)

    def tearDown(self):
        pass
