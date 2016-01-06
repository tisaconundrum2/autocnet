import pvl

VERSION = 2
HEADERSTARTBYTE = 65536
DEFAULTUSERNAME = 'AutoControlNetGeneration'

def to_isis(path, C, mode='w', version=VERSION,
            headerstartbyte=HEADERSTARTBYTE,
            networkId=None, targetname=None,
            description=None, username=DEFAULTUSERNAME):
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
            header = store.create_header(C, version, headerstartbyte, networkId,
                                targetname, description, username)
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

    def create_header(self, cnet, version, headerstartbyte,
                      networkid, targetname, description, username):
        """
        Create the PVL header object
        Parameters
        ----------
        cnet : C
               A control net object

        Returns
        -------

        """

        encoder = pvl.encoder.IsisCubeLabelEncoder

        headerbytes = 1
        pointsstartbyte = 1
        pointsbytes = 1

        header = pvl.PVLModule([
            ('Protobuffer',
            {'Core':{'HeaderStartByte':headerstartbyte,
                    'HeaderBytes':headerbytes,
                    'PointsStartByte':pointsstartbyte,
                    'PointsBytes':pointsbytes}}),
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


