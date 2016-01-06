import os
import unittest
from unittest.mock import Mock
import sys
sys.path.insert(0, os.path.abspath('..'))

import pvl

from .. import io_controlnetwork
from .. import ControlNetFileV0002 as cnf

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
        cnet.creationdate = 'None'
        cnet.modifieddate = 'None'
        io_controlnetwork.to_isis('test.net', cnet, mode='wb')

    def test_create_buffer_header(self):
        header_message_size = 45
        with open('test.net', 'rb') as f:
            f.seek(io_controlnetwork.HEADERSTARTBYTE)
            raw_header_message = f.read(header_message_size)
            header_protocol = cnf.ControlNetFileHeaderV0002()
            self.assertEqual('', header_protocol.networkId)
            self.assertEqual('', header_protocol.targetName)

    def test_create_pvl_header(self):
        pvl_header = pvl.load('test.net')
        npoints = find_in_dict(pvl_header, 'NumberOfPoints')
        self.assertEqual(75, npoints)
        mpoints = find_in_dict(pvl_header, 'NumberOfMeasures')
        self.assertEqual(621, mpoints)

    def tearDown(self):
        os.remove('test.net')