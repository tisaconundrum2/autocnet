import os
import unittest

from sqlalchemy.orm import session

from autocnet import get_data

import sys
sys.path.insert(0, os.path.abspath('..'))

from .. import io_db


class TestDataDB(unittest.TestCase):

    def test_setup_session(self):
        data_session = io_db.setup_db_session(get_data('data.db'))
        self.assertIsInstance(data_session, session.Session)