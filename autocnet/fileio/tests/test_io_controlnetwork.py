import os
from time import gmtime, strftime
import unittest
from unittest.mock import MagicMock
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
        ids = ['pt1','pt1', 'pt1', 'pt2', 'pt2']
        ptype = [2,2,2,2,2]
        serials = ['a', 'b', 'c', 'b', 'c']
        mtype = [2,2,2,2,2]

        multi_index = pd.MultiIndex.from_tuples(list(zip(ids, ptype, serials, mtype)),
                                    names=['Id', 'Type', 'Serial Number', 'Measure Type'])

        columns = ['Random Number']
        self.data_length = 5
        data = np.random.randn(self.data_length)

        self.creation_time =  strftime("%Y-%m-%d %H:%M:%S", gmtime())
        cnet = C(data, index=multi_index, columns=columns)

        io_controlnetwork.to_isis('test.net', cnet, mode='wb')



    def test_create_buffer_header(self):
        header_message_size = 83
        with open('test.net', 'rb') as f:
            f.seek(io_controlnetwork.HEADERSTARTBYTE)
            raw_header_message = f.read(header_message_size)
            header_protocol = cnf.ControlNetFileHeaderV0002()
            header_protocol.ParseFromString(raw_header_message)

            #Non-repeating
            self.assertEqual('None', header_protocol.networkId)
            self.assertEqual('None', header_protocol.targetName)
            self.assertEqual(io_controlnetwork.DEFAULTUSERNAME,
                             header_protocol.userName)
            self.assertEqual(self.creation_time,
                             header_protocol.created)
            self.assertEqual('None', header_protocol.description)
            self.assertEqual('Not modified', header_protocol.lastModified)

            #Repeating
            self.assertEqual([31, 23], header_protocol.pointMessageSizes)

    def test_create_point(self):
        with open('test.net', 'rb') as f:
            point_protocol = cnf.ControlPointFileEntryV0002()
            #self.assertTrue(False)

    def test_create_pvl_header(self):
        pvl_header = pvl.load('test.net')

        npoints = find_in_dict(pvl_header, 'NumberOfPoints')
        self.assertEqual(2, npoints)

        mpoints = find_in_dict(pvl_header, 'NumberOfMeasures')
        self.assertEqual(5, mpoints)

        points_bytes = find_in_dict(pvl_header, 'PointsBytes')
        self.assertEqual(54, points_bytes)

        points_start_byte = find_in_dict(pvl_header, 'PointsStartByte')
        self.assertEqual(65619, points_start_byte)



    def tearDown(self):
        os.remove('test.net')