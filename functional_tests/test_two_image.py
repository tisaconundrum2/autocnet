import os
import unittest

import pandas as pd
from scipy.misc import bytescale

from autocnet.examples import get_path
from autocnet.fileio.io_controlnetwork import to_isis
from autocnet.fileio.io_gdal import GeoDataset
from autocnet.graph.network import CandidateGraph
from autocnet.matcher import feature_extractor as fe
from autocnet.matcher.matcher import FlannMatcher
from autocnet.matcher.matcher import OutlierDetector


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
        basepath = os.path.dirname(adjacency)
        cg = CandidateGraph.from_adjacency_file(adjacency)
        self.assertEqual(2, cg.number_of_nodes())
        self.assertEqual(1, cg.number_of_edges())

        # Step: Extract image data and attribute nodes
        cg.extract_features(25)
        for node, attributes in cg.nodes_iter(data=True):
            self.assertIn(len(attributes['keypoints']), [24, 25, 26])

        # Step: Then apply a FLANN matcher
        fl = FlannMatcher()
        for node, attributes in cg.nodes_iter(data=True):
            fl.add(attributes['descriptors'], key=node)
        fl.train()

        for node, attributes in cg.nodes_iter(data=True):
            descriptors = attributes['descriptors']
            matches = fl.query(descriptors, node, k=3) #had to increase from 2 to test distance ratio test
            detectme = OutlierDetector()
            cg.add_matches(matches)

        # Step: And create a C object
        cnet = cg.to_cnet()

        # Step update the serial numbers
        nid_to_serial = {}
        for node, attributes in cg.nodes_iter(data=True):
            nid_to_serial[node] = self.serial_numbers[attributes['image_name']]

        cnet.replace({'nid': nid_to_serial}, inplace=True)

        # Step: Output a control network
        to_isis('TestTwoImageMatching.net', cnet, mode='wb',
                networkid='TestTwoImageMatching', targetname='Moon')

    def tearDown(self):
        try:
            os.path.remove('TestTwoImageMatching.net')
        except: pass
