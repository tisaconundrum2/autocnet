from collections import MutableMapping

import numpy as np
import pandas as pd
from scipy.misc import bytescale

from autocnet.fileio.io_gdal import GeoDataset
from autocnet.matcher import feature_extractor as fe
from autocnet.matcher import outlier_detector as od
from autocnet.cg.cg import convex_hull_ratio
from autocnet.utils.isis_serial_numbers import generate_serial_number
from autocnet.vis.graph_view import plot_node


class Node(dict, MutableMapping):
    """
    Attributes
    ----------

    image_name : str
                 Name of the image, with extension
    image_path : str
                 Relative or absolute PATH to the image
    handle : object
             File handle to the object
    keypoints : dataframe
                With columns, x, y, and response
    nkeypoints : int
                 The number of keypoints found for this image
    descriptors : ndarray
                  32-bit array of feature descriptors returned by OpenCV
    masks : set
            A list of the available masking arrays

    isis_serial : str
                  If the input images have PVL headers, generate an
                  ISIS compatible serial number

     provenance : dict
                  With key equal to an autoincrementing integer and value
                  equal to a dict of parameters used to generate this
                  realization.
    """

    def __init__(self, image_name=None, image_path=None):
        self.image_name = image_name
        self.image_path = image_path
        self._masks = set()
        self._mask_arrays = {}
        self.provenance = {}
        self._pid = 0

    def __repr__(self):
        return """
        NodeID: {}
        Image Name: {}
        Image PATH: {}
        Number Keypoints: {}
        Available Masks : {}
        """.format(None, self.image_name, self.image_path,
                   self.nkeypoints, self.masks)

    @property
    def handle(self):
        if not getattr(self, '_handle', None):
            self._handle = GeoDataset(self.image_path)
        return self._handle

    @property
    def nkeypoints(self):
        if hasattr(self, '_nkeypoints'):
            return self._nkeypoints
        else:
            return 0

    @nkeypoints.setter
    def nkeypoints(self, v):
        self._nkeypoints = v

    @property
    def masks(self):
        return self._masks

    @masks.setter
    def masks(self, v):
        self._masks.add(v[0])
        self._mask_arrays[v[0]] = v[1]

    def get_array(self, band=1):
        """
        Get a band as a 32-bit numpy array

        Parameters
        ----------
        band : int
               The band to read, default 1
        """

        array = self.handle.read_array(band=band)
        return bytescale(array)

    def extract_features(self, array, **kwargs):
        """
        Extract features for the node

        Parameters
        ----------
        array : ndarray

        kwargs : dict
                 KWargs passed to autocnet.feature_extractor.extract_features

        """
        keypoint_objs, descriptors = fe.extract_features(array, **kwargs)
        keypoints = np.empty((len(keypoint_objs), 7),dtype=np.float32)
        for i, kpt in enumerate(keypoint_objs):
            octave = kpt.octave & 8
            layer = (kpt.octave >> 8) & 255
            if octave < 128:
                octave = octave
            else:
                octave = (-128 | octave)
            keypoints[i] = kpt.pt[0], kpt.pt[1], kpt.response, kpt.size, kpt.angle, octave, layer  # y, x
        self.keypoints = pd.DataFrame(keypoints, columns=['x', 'y', 'response', 'size',
                                                          'angle', 'octave', 'layer'])
        self._nkeypoints = len(self.keypoints)
        self.descriptors = descriptors.astype(np.float32)

        self.provenance[self._pid] = {'detector': 'sift',
                                      'parameters':kwargs}
        self._pid += 1

    def anms(self, nfeatures=100, robust=0.9):
        mask = od.adaptive_non_max_suppression(self.keypoints,nfeatures,robust)
        self.masks = ('anms', mask)

    def coverage_ratio(self, clean_keys=[]):
        """
        Compute the ratio $area_{convexhull} / area_{total}$

        Returns
        -------
        ratio : float
                The ratio of convex hull area to total area.
        """
        ideal_area = self.handle.pixel_area
        if not hasattr(self, 'keypoints'):
            raise AttributeError('Keypoints must be extracted already, they have not been.')

        if clean_keys:
            mask = np.prod([self._mask_arrays[i] for i in clean_keys], axis=0, dtype=np.bool)
            keypoints = self.keypoints[mask]

        keypoints = self.keypoints[['x', 'y']].values

        ratio = convex_hull_ratio(keypoints, ideal_area)
        return ratio

    def plot(self, clean_keys=[], **kwargs):  # pragma: no cover
        return plot_node(self, clean_keys=clean_keys, **kwargs)

    @property
    def isis_serial(self):
        """
        Generate an ISIS compatible serial number using the data file
        associated with this node.  This assumes that the data file
        has a PVL header.
        """
        if not hasattr(self, '_isis_serial'):
            try:
                self._isis_serial = generate_serial_number(self.image_path)
            except:
                self._isis_serial = None
        return self._isis_serial