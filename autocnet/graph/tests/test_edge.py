import unittest
from unittest.mock import Mock

import pandas as pd

from .. import edge
from .. import node


class TestEdge(unittest.TestCase):

    def setUp(self):
        source = Mock(node.Node)
        destination = Mock(node.Node)
        self.edge = edge.Edge(source=source, destination=destination)

        '''
        # Define a matches dataframe
        source_image = np.zeros(20)
        destination_image = np.ones(20)
        source_idx = np.repeat(np.arange(10), 2)
        destination_idx = np.array([336,  78, 267, 467, 214, 212, 463, 241,  27, 154, 320, 108, 196,
                                    460,  67, 135,  80, 122, 106, 343])
        distance = np.array([263.43121338,  287.05050659,  231.03895569,  242.14459229,
                             140.07498169,  299.86331177,  332.05722046,  337.71438599,
                             94.9052124,  208.04806519,  102.21056366,  173.48774719,
                             102.19099426,  237.63206482,  240.93359375,  277.74627686,
                             217.82791138,  224.22979736,  260.3939209,  287.91143799])
        data = np.stack((source_image, source_idx, destination_image, destination_idx, distance), axis=-1)
        self.edge.matches = pd.DataFrame(data, columns=['source_image', 'source_idx',
                                                 'destination_image', 'destination_idx',
                                                 'distance'])
        '''

    def test_properties(self):
        pass

    def test_masks(self):
        self.assertIsInstance(self.edge.masks, pd.DataFrame)

    def test_health(self):
        self.assertEqual(self.edge.health, 1.0)

    def test_compute_fundamental_matrix(self):
        with self.assertRaises(AttributeError):
            self.edge.compute_fundamental_matrix()
