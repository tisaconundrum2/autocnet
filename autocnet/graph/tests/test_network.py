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


class TestFromList(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filelist = [get_path('Mars_MGS_MOLA_ClrShade_MAP2_0.0N0.0_MERC.tif'),
                    get_path('Lunar_LRO_LOLA_Shade_MAP2_90.0N20.0_LAMB.tif'),
                    get_path('Mars_MGS_MOLA_ClrShade_MAP2_90.0N0.0_POLA.tif')]
        cls.graph = network.CandidateGraph.from_filelist(filelist)

    def test_graph_length(self):
        self.assertEqual(self.graph.__len__(), 3)
        self.assertEqual(self.graph.number_of_nodes(), 3)

'''
class TestFromListCubes(unittest.TestCase):
    @classmethod

    def setUpClass(cls):
        filelist = [get_path('AS15-M-0297_sub4.cub'),
                    get_path('AS15-M-0298_sub4.cub'),
                    get_path('AS15-M-0299_sub4.cub')]
        cls.graph = network.CandidateGraph.from_filelist(filelist)

    def test_graph_length(self):
        self.assertEqual(self.graph.number_of_nodes(), 3)
        self.assertEqual(self.graph.number_of_edges(), 3)
'''


class TestEdge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = network.CandidateGraph.from_adjacency(get_path('adjacency.json'))


class TestMSTGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dict = {"0": ["4", "2", "1", "3"],
                         "1": ["0", "3", "2", "6", "5"],
                         "2": ["1", "0", "3", "4", "7"],
                         "3": ["2", "0", "1", "5"],
                         "4": ["2", "0"],
                         "5": ["1", "3"],
                         "6": ["1"],
                         "7": ["2"]}

        cls.graph = network.CandidateGraph.from_adjacency(cls.test_dict)
        cls.graph.minimum_spanning_tree()
        cls.mst_graph = cls.graph.copy()

        for s, d, edge in cls.graph.edges_iter(data=True):
            if not cls.graph.graph_masks['mst'][(s, d)]:
                cls.mst_graph.remove_edge(s, d)

    def test_mst_output(self):
        self.assertEqual(self.mst_graph.nodes(), self.graph.nodes())
        self.assertEqual(self.mst_graph.number_of_edges(), self.graph.number_of_edges()-5)