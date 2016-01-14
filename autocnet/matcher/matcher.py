import cv2
import numpy as np
import pandas as pd
from autocnet.graph.network import CandidateGraph

FLANN_INDEX_KDTREE = 1
DEFAULT_FLANN_PARAMETERS = dict(algorithm=FLANN_INDEX_KDTREE,
                                trees=3)


class FlannMatcher(object):
    """
    A wrapper to the OpenCV Flann based matcher class that adds
    metadata tracking attributes and methods.

    Attributes
    ----------
    image_indices : dict
                    with key equal to the train image idx (returned by the DMatch object)
                    and value equal to the image identifier, e.g. the name

    image_index_counter : int
                          The current number of images loaded into the matcher
    """

    def __init__(self, flann_parameters=DEFAULT_FLANN_PARAMETERS):
        self._flann_matcher = cv2.FlannBasedMatcher(flann_parameters, {})
        self.image_indices = {}
        self.image_index_counter = 0

    def add(self, descriptor, key):
        """
        Add a set of descriptors to the matcher and add the image
        index key to the image_indices attribute

        Parameters
        ----------
        descriptor : ndarray
                     The descriptor to be added

        key : hashable
              The identifier for this image, e.g. the image name
        """
        self._flann_matcher.add([descriptor])
        self.image_indices[self.image_index_counter] = key
        self.image_index_counter += 1

    def train(self):
        """
        Using the descriptors, generate the KDTree
        """
        self._flann_matcher.train()

    def query(self, descriptor, k=3):
        """

        Parameters
        ----------
        descriptor : ndarray
                     The query descriptor to search for

        k : int
            The number of nearest neighbors to search for

        Returns
        -------
        matched : dataframe
                  containing matched points
        """
        matches = self._flann_matcher.knnMatch(descriptor, k=k)
        matched = []
        for m in matches:
            matched.append((self.image_indices[m[1].imgIdx],
                  m[1].queryIdx,
                  m[1].trainIdx,
                  m[1].distance))
        return pd.DataFrame(matched, columns=['matched_to', 'queryIdx',
                                              'trainIdx', 'distance'])

