import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import cv2
import numpy as np
import unittest

from autocnet.examples import get_path

from .. import network


class TestCandidateGraph(unittest.TestCase):
    
    def setUp(self):
        self.graph = network.CandidateGraph.from_adjacency_file(get_path('adjacency.json'))

    def test_get_name(self):
        node_number = self.graph.node_name_map['AS15-M-0297_SML.png']
        name = self.graph.get_name(node_number)
        self.assertEquals(name, 'AS15-M-0297_SML.png')

    def test_add_image(self):
        with self.assertRaises(NotImplementedError):
            self.graph.add_image()

    def test_to_json_file(self):
        self.graph.to_json_file('test_graph_to_json.json')
        self.assertTrue(os.path.exists('test_graph_to_json.json'))

    def test_extract_features(self):
        # also tests get_geodataset() and get_keypoints
        self.graph.extract_features(10)
        node_number = self.graph.node_name_map['AS15-M-0297_SML.png']
        node = self.graph.node[node_number]
        self.assertEquals(len(node['image']), 1012)
        self.assertEquals(len(node['keypoints']), 10)
        self.assertEquals(len(node['descriptors']), 10)
        self.assertIsInstance(node['keypoints'][0], type(cv2.KeyPoint()))
        self.assertIsInstance(node['descriptors'][0], np.ndarray)
        self.assertEquals(self.graph.get_keypoints(node_number), node['keypoints'])


    def tearDown(self):
        try:
            os.remove('test_graph_to_json.json')
        except:
            pass
