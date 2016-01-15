import os
import numpy as np
import unittest
from autocnet.examples import get_path
import cv2

import sys
sys.path.insert(0, os.path.abspath('..'))

from .. import feature_extractor

from autocnet.fileio import io_gdal

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.dataset = io_gdal.GeoDataset(get_path('Mars_MGS_MOLA_ClrShade_MAP2_0.0N0.0_MERC.tif'))
        self.data_array = self.dataset.read_array()

    def test_extract_features(self):
        features = feature_extractor.extract_features(self.data_array, 10)
        self.assertEquals(len(features), 2)
        self.assertEqual(len(features[0]), 11)  # OpenCV +1 to
        self.assertIsInstance(features[0][0], type(cv2.KeyPoint()))
        self.assertIsInstance(features[1][0], np.ndarray)

