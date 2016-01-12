import sys

import pvl
import numpy as np

from autocnet.fileio import ControlNetFileV0002_pb2 as cnf

#TODO: Protobuf3 should be a conditional import, if availble use it, otherwise bail

VERSION = 2
HEADERSTARTBYTE = 65536
DEFAULTUSERNAME = 'AutoControlNetGeneration'

def to_isis(path, C, mode='w', version=VERSION,
            headerstartbyte=HEADERSTARTBYTE,
            networkid='None', targetname='None',
            description='None', username=DEFAULTUSERNAME):
    """
    Parameters
    ----------
    path : str
           Input path where the file is to be written

    C : object
           A control network object

    mode : {'a', 'w', 'r', 'r+'}

        ``'r'``
            Read-only; no data can be modified.
        ``'w'``
            Write; a new file is created (an existing file with the same
            name would be deleted).
        ``'a'``
            Append; an existing file is opened for reading and writing,
            and if the file does not exist it is created.
        ``'r+'``
            It is similar to ``'a'``, but the file must already exist.

    version : int
          The current ISIS version to write, defaults to 2

    headerstartbyte : int
                      The seek offset that the protocol buffer header starts at

    networkid : str
                The name of the network

    targetname : str
                 The name of the target, e.g. Moon

    description : str
                  A description for the network.

    username : str
               The name of the user / application that created the control network
"""

    if isinstance(path, str):
        with IsisStore(path, mode) as store:
            point_messages, point_sizes = store.create_points(C)
            points_bytes = sum(point_sizes)
            #store.write()
            buffer_header, buffer_header_size = store.create_buffer_header(C, networkid,
                                                                           targetname,
                                                                           description,
                                                                           username,
                                                                           point_sizes)

            store.write(buffer_header,HEADERSTARTBYTE)

            header = store.create_pvl_header(C, version, headerstartbyte, networkid,
                                             targetname, description, username,
                                             buffer_header_size, points_bytes)


            store.write(header)

class IsisStore(object):
    """
    Class to manage IO of an ISIS3 control network (version 2).
    """

    def __init__(self, path, mode=None, **kwargs):
        self._path = path
        if not mode:
            mode = 'a'
        self._mode = mode
        self._handle = None

        self._open()

    def _open(self):
        if self._mode in ['wb', 'a']:
            self._handle = open(self._path, self._mode)
        else:
            raise NotImplementedError

    def write(self, data, offset=0):
        """
        Parameters
        ----------
        C : object
               A control network object
        """
        self._handle.seek(offset)
        self._handle.write(data)

    def create_points(self, cnet):
        point_sizes = []
        point_messages = []

        for point_id in cnet.index.levels[0]:

            # Instantiate the proto spec
            point_spec = cnf.ControlPointFileEntryV0002()

            # Get the subset of the dataframe
            point = cnet.loc[point_id]

            point_spec.id = point_id
            point_spec.type = 2  # Hard coded to free

            # A single extend call is cheaper than many add calls to pack points
            measure_iterable = []

            for m in point.iterrows():
                measure_spec = point_spec.Measure()
                serialnumber = m[0][1]
                mtype = m[0][2]
                measure_spec.serialnumber = serialnumber
                measure_spec.type = mtype
                measure_iterable.append(measure_spec)
            point_spec.measures.extend(measure_iterable)

            point_message = point_spec.SerializeToString()
            point_sizes.append(sys.getsizeof(point_message))
            point_messages.append(point_message)

        return point_messages, point_sizes

    def create_buffer_header(self, cnet, networkid, targetname,
                             description, username, point_sizes):
        """
        Create the Google Protocol Buffer header using the
        protobuf spec.

        Parameters
        ----------
        cnet : object
               A control network object

        Returns
        -------
        message : str
                  The serialized message to write
        """
        raw_header_message = cnf.ControlNetFileHeaderV0002()
        raw_header_message.created = cnet.creationdate
        raw_header_message.lastModified = cnet.modifieddate
        raw_header_message.networkId = networkid
        raw_header_message.description = description
        raw_header_message.targetName = targetname
        raw_header_message.userName = username

        raw_header_message.pointMessageSizes.extend(point_sizes)
        header_message = raw_header_message.SerializeToString()
        header_message_size = sys.getsizeof(header_message)

        return header_message, header_message_size

    def create_pvl_header(self, cnet, version, headerstartbyte,
                      networkid, targetname, description, username,
                          buffer_header_size, points_bytes):
        """
        Create the PVL header object
        Parameters
        ----------
        cnet : C
               A control net object

        points_bytes : int
                       The total number of bytes all points require

        Returns
        -------

        """

        encoder = pvl.encoder.IsisCubeLabelEncoder

        header_bytes = buffer_header_size
        points_start_byte = HEADERSTARTBYTE + buffer_header_size

        header = pvl.PVLModule([
            ('Protobuffer',
            {'Core':{'HeaderStartByte':headerstartbyte,
                    'HeaderBytes':header_bytes,
                    'PointsStartByte':points_start_byte,
                    'PointsBytes':points_bytes}}),
            ('ControlNetworkInfo',pvl.PVLGroup([
                    ('NetworkId', networkid),
                    ('TargetName', targetname),
                    ('UserName', username),
                    ('Created',cnet.creationdate),
                    ('LastModified',cnet.modifieddate),
                    ('Description',description),
                    ('NumberOfPoints',cnet.n),
                    ('NumberOfMeasures',cnet.m),
                    ('Version',version)
                    ]))
        ])
        return pvl.dumps(header, cls=encoder)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    def close(self):
        if self._handle is not None:
            self._handle.close()
        self._handle = None


