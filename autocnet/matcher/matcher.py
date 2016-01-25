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

    def query(self, descriptor, query_image, k=3):
        """

        Parameters
        ----------
        descriptor : ndarray
                     The query descriptor to search for

        query_image : hashable
                      Key of the query image

        k : int
            The number of nearest neighbors to search for

        Returns
        -------
        matched : dataframe
                  containing matched points with columns containing:
                  matched image name, query index, train index, and
                  descriptor distance
        """

        matches = self._flann_matcher.knnMatch(descriptor, k=k)
        matched = []
        for m in matches:
            for i in m:
                # This checks for self neighbor and never allows them into the graph
                if self.image_indices[i.imgIdx] == query_image:
                    continue

                # Ensure ordering in the source / destination
                if query_image < self.image_indices[i.imgIdx]:
                    matched.append((query_image,
                                    i.queryIdx,
                                    self.image_indices[i.imgIdx],
                                    i.trainIdx,
                                    i.distance))
                else:
                    matched.append((self.image_indices[i.imgIdx],
                                    i.trainIdx,
                                    query_image,
                                    i.queryIdx,
                                    i.distance))
        return pd.DataFrame(matched, columns=['source_image', 'source_idx',
                                              'destination_image', 'destination_idx',
                                              'distance'])

#TODO: decide on a consistent mask format to output.
#Do we want to also accept existing masks and just mask more things?
#consider passing in the matches and source_node to __init__
class MatchOutlierDetector(object):
    """
    Documentation
    """
    def __init__(self, matches, ratio=0.8):
        #0.8 is Lowe's paper value -- can be changed.
        self.distance_ratio = ratio
        self.matches = matches
        self.mask = None #start with empty mask? I guess we could accept an input mask.

    # return mask with self-neighbors set to zero. (query only takes care of literal self-matches on a keypoint basis, not self-matches for the whole image)
    #TODO: turn this into a mask-style thing. just returns a mask of bad values
    def self_neighbors(self, source_node):
        """
        Returns a df containing self-neighbors that must be removed.
        (temporary return val?)

        Parameters
        ----------
        matches : dataframe
                  The pandas dataframe output by FlannMatcher.query()
                  containing matched points with columns containing:
                  matched image name, query index, train index, and
                  descriptor distance

        source_node: a string used as the key of the matched node

        Returns
        -------
        """
        mask = []
        self_matches = self.matches.loc[self.matches['matched_to'] == source_node]
        print(self_matches)
        return mask
        #this could maybe be return maches.source_node == matches.destination_node

    #also add a mirroring(?) test?

    def distance_ratio(self):
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
        #mask = []
        mask = {}
        for key, group in self.matches.groupBy('queryIdx'):
            #won't work if there's only 1 match for each queryIdx
            if len(group) < 2:
                pass #actually need to make sure that none of these are masked.
            else:
                if group['distance'].iloc[0] < self.distance_ratio * group['distance'].iloc[1]:
                    mask.append([1])
                else:
                    mask.append([0])
        return mask
         #make the mask a dict between indicies of the original df (if possible) and true/false values!