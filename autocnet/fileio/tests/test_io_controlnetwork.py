import os
from time import gmtime, strftime
import unittest
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import pvl

from .. import io_controlnetwork
from .. import ControlNetFileV0002_pb2 as cnf

from autocnet.utils.utils import find_in_dict
from autocnet.control.control import C

class TestWriteIsisControlNetwork(unittest.TestCase):

    def setUp(self):
        """
        Not 100% sure how to mock in the DF without creating lots of methods...
        """

        serial_times = {295: '1971-07-31T01:24:11.754',
                   296: '1971-07-31T01:24:36.970',
                   297: '1971-07-31T01:25:02.243',
                   298: '1971-07-31T01:25:27.457',
                   299: '1971-07-31T01:25:52.669',
                   300: '1971-07-31T01:26:17.923'}
        self.serials = ['APOLLO15/METRIC/{}'.format(i) for i in serial_times.values()]

        ids = ['pt1','pt1', 'pt1', 'pt2', 'pt2']
        ptype = [2, 2, 2, 2, 2]
        serials = [self.serials[0], self.serials[1], self.serials[2],
                   self.serials[2], self.serials[3]]
        mtype = [2, 2, 2, 2, 2]

        multi_index = pd.MultiIndex.from_tuples(list(zip(ids, ptype, serials, mtype)),
                                    names=['Id', 'Type', 'Serial Number', 'Measure Type'])

        columns = ['Random Number']
        self.data_length = 5
        data = np.random.randn(self.data_length)

        self.creation_time =  strftime("%Y-%m-%d %H:%M:%S", gmtime())
        cnet = C(data, index=multi_index, columns=columns)

        io_controlnetwork.to_isis('test.net', cnet, mode='wb', targetname='Moon')

        self.header_message_size = 84
        self.point_start_byte = 65620

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
            self.assertEqual(self.creation_time,
                             header_protocol.created)
            self.assertEqual('None', header_protocol.description)
            self.assertEqual('Not modified', header_protocol.lastModified)

            #Repeating
            self.assertEqual([145, 99], header_protocol.pointMessageSizes)

    def test_create_point(self):
        with open('test.net', 'rb') as f:

            with open('test.net', 'rb') as f:
                f.seek(self.point_start_byte)
                for i, length in enumerate([145, 99]):
                    point_protocol = cnf.ControlPointFileEntryV0002()
                    raw_point = f.read(length)
                    point_protocol.ParseFromString(raw_point)
                    self.assertEqual('pt{}'.format(i+1), point_protocol.id)
                    self.assertEqual(2, point_protocol.type)
                    for m in point_protocol.measures:
                        self.assertTrue(m.serialnumber in self.serials)
                        self.assertEqual(2, m.type)

    def test_create_pvl_header(self):
        pvl_header = pvl.load('test.net')

        npoints = find_in_dict(pvl_header, 'NumberOfPoints')
        self.assertEqual(2, npoints)

        mpoints = find_in_dict(pvl_header, 'NumberOfMeasures')
        self.assertEqual(5, mpoints)

        points_bytes = find_in_dict(pvl_header, 'PointsBytes')
        self.assertEqual(244, points_bytes)

        points_start_byte = find_in_dict(pvl_header, 'PointsStartByte')
        self.assertEqual(65620, points_start_byte)

    def tearDown(self):
        os.remove('test.net')