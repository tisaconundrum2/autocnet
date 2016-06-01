import unittest
import ogr

from unittest.mock import Mock
from unittest.mock import MagicMock

from autocnet.fileio import io_gdal
from autocnet.examples import get_path
from autocnet.graph.network import CandidateGraph
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

    def test_edge_overlap(self):
        e = edge.Edge()
        e.weight = {}
        source = Mock(spec = node.Node)
        destination = Mock(spec = node.Node)
        e.destination = destination
        e.source = source
        geodata_s = Mock(spec = io_gdal.GeoDataset)
        geodata_d = Mock(spec = io_gdal.GeoDataset)
        source.geodata = geodata_s
        destination.geodata = geodata_d

        wkt1 = "POLYGON ((0 40, 40 40, 40 0, 0 0, 0 40))"
        wkt2 = "POLYGON ((20 60, 60 60, 60 20, 20 20, 20 60))"

        poly1 = ogr.CreateGeometryFromWkt(wkt1)
        poly2 = ogr.CreateGeometryFromWkt(wkt2)

        source.geodata.footprint = poly1
        destination.geodata.footprint = poly2

        e.overlap()
        self.assertEqual(e.weight['overlap_area'], 400)
        self.assertAlmostEqual(e.weight['overlap_percn'], 14.285714285)

    def test_coverage(self):
        df1 = pd.DataFrame({'x': (15, 18, 18, 12, 12), 'y': (5, 10, 15, 15, 10)})
        array1 = [True, True, True, True, True]
        array2 = [[0, 0, 1, 0],
                  [0, 1, 1, 1],
                  [0, 2, 1, 2],
                  [0, 3, 1, 3],
                  [0, 4, 1, 4]]
        df2 = pd.DataFrame(data = array1, columns = ['symmetry'], dtype = bool)
        df3 = pd.DataFrame(data = array2, columns = ['source_image', 'source_idx', 'destination_image', 'destination_idx'] )
        e = Mock(spec = edge.Edge())
        source_node = node.Node()
        destination_node = node.Node()

        source_node.get_keypoint_coordinates = MagicMock(return_value=df1)
        destination_node.get_keypoint_coordinates = MagicMock(return_value=df1)

        e.source = source_node
        e.destination = destination_node

        e.matches = df3
        # e.masks = df2

        e.coverage(image = 'source')