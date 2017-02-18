import unittest

import numpy as np
import networkx as nx

from .. import markov_cluster

from autocnet.examples import get_path
from autocnet.graph.network import CandidateGraph


class TestMarkovCluster(unittest.TestCase):

    def setUp(self):
        pass
        #TODO: These tests need to load a graph from the new zip
        #self.g = CandidateGraph.from_graph(get_path('sixty_four_apollo.graph'))

    def test_mcl_from_network(self):
        pass
        #self.g.compute_clusters(inflate_factor=15)
        #self.assertIsInstance(self.g.clusters, dict)
        #self.assertEqual(len(self.g.clusters), 14)

    def test_mcl_from_adj_matrix(self):
        pass
        #arr = np.array(nx.adjacency_matrix(self.g).todense())
        #flow, clusters = markov_cluster.mcl(arr)
        #self.assertIsInstance(clusters, dict)
        #self.assertEqual(len(clusters), 3)
