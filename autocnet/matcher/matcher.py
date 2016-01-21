import cv2
import numpy as np
import pandas as pd
from autocnet.graph.network import CandidateGraph


FLANN_INDEX_KDTREE = 1  # Algorithm to set centers,
DEFAULT_FLANN_PARAMETERS = dict(algorithm=FLANN_INDEX_KDTREE,
                                trees=3)

class FlannMatcher(object):
    """
    A wrapper to the OpenCV Flann based matcher class that adds
    metadata tracking attributes and methods.  This takes arbitrary
    descriptors and so should be available for use with any
    descriptor data stored as an ndarray.

    Attributes
    ----------
    image_indices : dict
                    with key equal to the train image index (returned by the DMatch object),
                    e.g. an integer array index
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

    #consider changing this back to returning (matches, data_frame)
    def query(self, descriptor, k=3, self_neighbor=True):
        """

        Parameters
        ----------
        descriptor : ndarray
                     The query descriptor to search for

        k : int
            The number of nearest neighbors to search for

        self_neighbor : bool
                        If the query descriptor is also a member
                        of the KDTree avoid self neighbor, default True.

        Returns
        -------
        matched : dataframe
                  containing matched points with columns containing:
                  matched image name, query index, train index, and
                  descriptor distance
        """
        idx = 0
        if self_neighbor:
            idx = 1
        matches = self._flann_matcher.knnMatch(descriptor, k=k)
        matched = []
        for m in matches:
            for i in m[idx:]:
                matched.append((self.image_indices[i.imgIdx],
                                i.queryIdx,
                                i.trainIdx,
                                i.distance))
        data_frame = pd.DataFrame(matched, columns=['matched_to', 'queryIdx',
                                              'trainIdx', 'distance'])
        return data_frame

#don't throw anything out, just have dataframes and masks
#TODO: decide on a consistent mask format to output. Do we want to also accept existing masks and just mask more things?
class MatchOutlierDetector(object):
    """
    Documentation
    """

    def __init__(self, ratio=0.8):
        #0.8 is Lowe's paper value -- can be changed.
        self.distance_ratio = ratio

    # return mask with self-neighbors set to zero. (query only takes care of literal self-matches on a keypoint basis, not self-matches for the whole image)
    # matches: a dataframe
    # source_node: a string with the name of the node that was just matched.
    #TODO: turn this into a mask-style thing. just returns a mask of bad values
    def find_self_neighbors(self, source_node, matches):
        mask = []
#        filtered_matches = matches.loc[matches['matched_to'] != source_node]
        self_matches = matches.loc[matches['matched_to'] == source_node]
        return mask

    # return mask with nodes that fail the distance ratio test set to zero
    #TODO: make more SQL-y / actually use dfs as expected
    # matches : dataframe
    def distance_ratio_test(self, matches):
        """
        Compute and return a mask for the matches dataframe returned by FlannMatcher.query()
        using the ratio test and distance_ratio set during initialization.

        Parameters
        ----------
        matches : dataframe
                  The pandas dataframe output by FlannMatcher.query()
                  containing matched points with columns containing:
                  matched image name, query index, train index, and
                  descriptor distance

        Returns
        -------
        mask : list
               a list of the same size as the matches dataframe
               with value = [1] if that entry in the df should be included
               and [0] if that entry in the df should be excluded
        """
        mask = []
        for key, group in matches.groupBy('queryIdx'):
            #won't work if there's only 1 match for each queryIdx
            if len(group) < 2:
                pass
            else:
                if group['distance'].iloc[0] < self.distance_ratio * group['distance'].iloc[1]:
                    mask.append([1])
                else:
                    mask.append([0])
        return mask
