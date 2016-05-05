import os
import sys
from time import gmtime, strftime
import unittest

from unittest.mock import Mock, MagicMock

from autocnet.graph.edge import Edge
from autocnet.graph.node import Node

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))

from autocnet.control import control


class TestC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        npts = 10
        coords = pd.DataFrame(np.arange(npts * 2).reshape(-1, 2))
        source = np.zeros(npts)
        destination = np.ones(npts)
        pid = np.arange(npts)

        matches = pd.DataFrame(np.vstack((source, pid, destination, pid)).T, columns=['source_image',
                                                                                      'source_idx',
                                                                                      'destination_image',
                                                                                      'destination_idx'])

        edge = Mock(spec=Edge)
        edge.source = Mock(spec=Node)
        edge.destination = Mock(spec=Node)
        edge.source.isis_serial = None
        edge.destination.isis_serial = None
        edge.source.get_keypoint_coordinates = MagicMock(return_value=coords)
        edge.destination.get_keypoint_coordinates = MagicMock(return_value=coords)

        cls.C = control.CorrespondenceNetwork()
        cls.C.add_correspondences(edge, matches)


    def test_n_point(self):
        self.assertEqual(self.C.n_points, 10)

    def test_n_measures(self):
        self.assertEqual(self.C.n_measures, 20)

    def test_modified_date(self):
        self.assertIsInstance(self.C.modifieddate, str)

    def test_creation_date(self):
        self.assertEqual(self.C.creationdate, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
