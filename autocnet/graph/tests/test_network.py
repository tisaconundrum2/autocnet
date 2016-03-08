import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import unittest

import numpy as np

from autocnet.examples import get_path

from .. import network


class TestCandidateGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        basepath = get_path('Apollo15')
        cls.graph = network.CandidateGraph.from_adjacency(get_path('three_image_adjacency.json'),
                                                          basepath=basepath)
        cls.disconnected_graph = network.CandidateGraph.from_adjacency(get_path('adjacency.json'))

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
        self.assertEqual(len(self.disconnected_graph.island_nodes()), 1)

    def test_connected_subgraphs(self):
        subgraph_list = self.disconnected_graph.connected_subgraphs()
        self.assertEqual(len(subgraph_list), 2)
        islands = self.disconnected_graph.island_nodes()
        self.assertTrue(islands[0] in subgraph_list[1])

        subgraph_list = self.graph.connected_subgraphs()
        self.assertEqual(len(subgraph_list), 1)


    def test_save_load(self):
        self.graph.save('test_save.cg')
        loaded = self.graph.from_graph('test_save.cg')
        self.assertEqual(self.graph.node[0].nkeypoints, loaded.node[0].nkeypoints)
        self.assertEqual(self.graph.edge[0][1], loaded.edge[0][1])

        a = self.graph.node[0].handle.read_array()
        b = loaded.node[0].handle.read_array()
        np.testing.assert_array_equal(a, b)

        os.remove('test_save.cg')

    def tearDown(self):
        pass


class TestEdge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = network.CandidateGraph.from_adjacency(get_path('adjacency.json'))

