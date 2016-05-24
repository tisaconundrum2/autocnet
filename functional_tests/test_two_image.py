import os
import ogr

import unittest
from unittest.mock import Mock

from autocnet.fileio import io_gdal
from autocnet.graph import edge
from autocnet.graph import node
from autocnet.examples import get_path
from autocnet.fileio.io_controlnetwork import to_isis
from autocnet.fileio.io_controlnetwork import write_filelist
from autocnet.graph.network import CandidateGraph


class TestTwoImageMatching(unittest.TestCase):
    """
    Feature: As a user
        I wish to automatically match two images to
        Generate an ISIS control network

        Scenario: Match two images
            Given a manually specified adjacency structure named two_image_adjacency.json
            When read create an adjacency graph
            Then extract image data and attribute nodes
            And find features and descriptors
            And apply a FLANN matcher
            Then create a C object from the graph matches
            Then output a control network
    """

    def setUp(self):
        self.serial_numbers = {'AS15-M-0295_SML.png': '1971-07-31T01:24:11.754',
                               'AS15-M-0296_SML.png': '1971-07-31T01:24:36.970',
                               'AS15-M-0297_SML.png': '1971-07-31T01:25:02.243',
                               'AS15-M-0298_SML.png': '1971-07-31T01:25:27.457',
                               'AS15-M-0299_SML.png': '1971-07-31T01:25:52.669',
                               'AS15-M-0300_SML.png': '1971-07-31T01:26:17.923'}

        for k, v in self.serial_numbers.items():
            self.serial_numbers[k] = 'APOLLO15/METRIC/{}'.format(v)

    def test_two_image(self):
        # Step: Create an adjacency graph
        adjacency = get_path('two_image_adjacency.json')
        basepath = get_path('Apollo15')
        cg = CandidateGraph.from_adjacency(adjacency, basepath=basepath)
        self.assertEqual(2, cg.number_of_nodes())
        self.assertEqual(1, cg.number_of_edges())

        # Step: Extract image data and attribute nodes
        cg.extract_features(method='sift', extractor_parameters={"nfeatures":500})
        for i, node in cg.nodes_iter(data=True):
            self.assertIn(node.nkeypoints, range(490, 511))

        #Step: Compute the coverage ratios
        truth_ratios = [0.95351579,
                        0.93595664]
        for i, node in cg.nodes_iter(data=True):
            ratio = node.coverage_ratio()
            self.assertIn(round(ratio,8), truth_ratios)

        cg.match_features(k=2)

        # Perform the symmetry check
        cg.symmetry_checks()
        # Perform the ratio check
        cg.ratio_checks(clean_keys = ['symmetry'])
        # Create fundamental matrix
        cg.compute_fundamental_matrices(clean_keys = ['symmetry', 'ratio'])

        for source, destination, edge in cg.edges_iter(data=True):

            # Perform the symmetry check
            self.assertIn(edge.masks['symmetry'].sum(), range(400, 600))
            # Perform the ratio test
            self.assertIn(edge.masks['ratio'].sum(), range(30, 100))

            # Range needs to be set
            self.assertIn(edge.masks['fundamental'].sum(), range(20, 100))


        # Step: Compute the homographies and apply RANSAC
        cg.compute_homographies(clean_keys=['symmetry', 'ratio'])

        # Step: Compute the overlap ratio and coverage ratio
        for s, d, edge in cg.edges_iter(data=True):
            edge.coverage_ratio(clean_keys=['symmetry', 'ratio'])

        # Step: Compute subpixel offsets for candidate points
        cg.subpixel_register(clean_keys=['ransac'])

        # Step:
        cg.suppress()

        # Step: And create a C object
        cnet = cg.generate_cnet(clean_keys=['symmetry', 'ratio', 'ransac', 'subpixel'])

        # Step: Create a fromlist to go with the cnet and write it to a file
        filelist = cg.to_filelist()
        write_filelist(filelist, path="fromlis.lis")


        # Step: Output a control network
        to_isis('TestTwoImageMatching.net', cg.cn, mode='wb',
                networkid='TestTwoImageMatching', targetname='Moon')

    def tearDown(self):
        try:
            os.remove('TestTwoImageMatching.net')
            os.remove('fromlist.lis')
        except: pass

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
        self.assertEqual(e.weight['overlap_percn'], 14.285714285714285)
