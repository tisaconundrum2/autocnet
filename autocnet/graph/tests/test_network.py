import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import unittest

from autocnet.examples import get_path
from autocnet.fileio.io_gdal import GeoDataset

from .. import network


class TestCandidateGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = network.CandidateGraph.from_adjacency(get_path('adjacency.json'))

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
        try:
            os.remove('test_graph_to_json.json')
        except:
            pass

    def test_island_nodes(self):
        self.assertEqual(len(self.graph.island_nodes()), 1)

    def test_connected_subgraphs(self):
        subgraph_list = self.graph.connected_subgraphs()
        self.assertEqual(len(subgraph_list), 2)
        island = self.graph.island_nodes()[0]
        self.assertTrue(island in subgraph_list[1])

    def tearDown(self):
        pass


class TestEdge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = network.CandidateGraph.from_adjacency(get_path('adjacency.json'))


