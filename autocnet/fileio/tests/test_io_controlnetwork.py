import os
import sys
import unittest
from unittest.mock import Mock, MagicMock
import numpy as np
import pandas as pd
import pvl

from .. import io_controlnetwork
from .. import ControlNetFileV0002_pb2 as cnf

from autocnet.utils.utils import find_in_dict
from autocnet.control.control import CorrespondenceNetwork
from autocnet.graph.edge import Edge
from autocnet.graph.node import Node

sys.path.insert(0, os.path.abspath('..'))


class TestWriteIsisControlNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        serial_times = {295: '1971-07-31T01:24:11.754',
                        296: '1971-07-31T01:24:36.970'}
        cls.serials = ['APOLLO15/METRIC/{}'.format(i) for i in serial_times.values()]

        # Create an edge and a set of matches
        cls.npts = 5
        coords = pd.DataFrame(np.arange(cls.npts * 2).reshape(-1, 2))
        source = np.zeros(cls.npts)
        destination = np.ones(cls.npts)
        pid = np.arange(cls.npts)

        matches = pd.DataFrame(np.vstack((source, pid, destination, pid)).T, columns=['source_image',
                                                                                      'source_idx',
                                                                                      'destination_image',
                                                                                      'destination_idx'])

        edge = Mock(spec=Edge)
        edge.source = Mock(spec=Node)
        edge.destination = Mock(spec=Node)
        edge.source.isis_serial = cls.serials[0]
        edge.destination.isis_serial = cls.serials[1]
        edge.source.get_keypoint_coordinates = MagicMock(return_value=coords)
        edge.destination.get_keypoint_coordinates = MagicMock(return_value=coords)

        cnet = CorrespondenceNetwork()
        cnet.add_correspondences(edge, matches)
        cls.creation_date = cnet.creationdate
        cls.modified_date = cnet.modifieddate
        io_controlnetwork.to_isis('test.net', cnet, mode='wb', targetname='Moon')

        cls.header_message_size = 98
        cls.point_start_byte = 65634

    def test_create_buffer_header(self):
        with open('test.net', 'rb') as f:
            f.seek(io_controlnetwork.HEADERSTARTBYTE)
            raw_header_message = f.read(self.header_message_size)
            header_protocol = cnf.ControlNetFileHeaderV0002()
            header_protocol.ParseFromString(raw_header_message)

            #Non-repeating
            self.assertEqual('None', header_protocol.networkId)
            self.assertEqual('Moon', header_protocol.targetName)
            self.assertEqual(io_controlnetwork.DEFAULTUSERNAME,
                             header_protocol.userName)
            self.assertEqual(self.creation_date,
                             header_protocol.created)
            self.assertEqual('None', header_protocol.description)
            self.assertEqual(self.modified_date, header_protocol.lastModified)

            #Repeating
            self.assertEqual([135] * self.npts, header_protocol.pointMessageSizes)

    def test_create_point(self):
        with open('test.net', 'rb') as f:

            with open('test.net', 'rb') as f:
                f.seek(self.point_start_byte)
                for i, length in enumerate([135] * self.npts):
                    point_protocol = cnf.ControlPointFileEntryV0002()
                    raw_point = f.read(length)
                    point_protocol.ParseFromString(raw_point)
                    self.assertEqual(str(i), point_protocol.id)
                    self.assertEqual(2, point_protocol.type)
                    for m in point_protocol.measures:
                        self.assertTrue(m.serialnumber in self.serials)
                        self.assertEqual(2, m.type)

    def test_create_pvl_header(self):
        pvl_header = pvl.load('test.net')

        npoints = find_in_dict(pvl_header, 'NumberOfPoints')
        self.assertEqual(5, npoints)

        mpoints = find_in_dict(pvl_header, 'NumberOfMeasures')
        self.assertEqual(10, mpoints)

        points_bytes = find_in_dict(pvl_header, 'PointsBytes')
        self.assertEqual(675, points_bytes)

        points_start_byte = find_in_dict(pvl_header, 'PointsStartByte')
        self.assertEqual(65634, points_start_byte)

    @classmethod
    def tearDownClass(cls):
        os.remove('test.net')
