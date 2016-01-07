import os
from time import gmtime, strftime
import unittest
from unittest.mock import Mock
import sys
sys.path.insert(0, os.path.abspath('..'))

import pvl

from .. import io_controlnetwork
from .. import ControlNetFileV0002_pb2 as cnf

from autocnet.utils.utils import find_in_dict
from autocnet.control.control import C

class TestWriteIsisControlNetwork(unittest.TestCase):

    def setUp(self):
        """
        The C object is mocked in, we do not want to test it here
        """
        cnet = Mock(spec=C)
        cnet.n = 75
        cnet.m = 621
        self.creation_time =  strftime("%Y-%m-%d %H:%M:%S", gmtime())
        cnet.creationdate = self.creation_time
        self.modified = 'Not modified'
        cnet.modifieddate = self.modified
        io_controlnetwork.to_isis('test.net', cnet, mode='wb')

    def test_create_buffer_header(self):
        header_message_size = 124
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
            self.assertEqual(self.modified, header_protocol.lastModified)

            #Repeating
            self.assertEqual(list(range(10)), header_protocol.pointMessageSizes)

    def test_create_pvl_header(self):
        pvl_header = pvl.load('test.net')
        npoints = find_in_dict(pvl_header, 'NumberOfPoints')
        self.assertEqual(75, npoints)
        mpoints = find_in_dict(pvl_header, 'NumberOfMeasures')
        self.assertEqual(621, mpoints)

    def tearDown(self):
        os.remove('test.net')