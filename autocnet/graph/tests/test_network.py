import os
import unittest

from autocnet.examples import get_path

import sys
sys.path.insert(0, os.path.abspath('..'))

from .. import network


class TestCandidateGraph(unittest.TestCase):
    
    def setUp(self):
        self.graph = network.CandidateGraph.from_adjacency(get_path('adjacency.json'))

    def test_add_image(self):
        self.graph.add_image()
        self.assertEqual(self.graph.node_counter, 7)

    def test_adjacency_to_json(self):
        self.graph.adjacency_to_json('test_adjacency_to_json.json')
        self.assertTrue(os.path.exists('test_adjacency_to_json.json'))

    def tearDown(self):
        try:
            os.remove('test_adjacency_to_json.json')
        except:
            pass
