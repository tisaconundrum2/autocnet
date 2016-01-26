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

class OutlierDetector(object):
    """
    A class which contains several outlier detection methods which all return
    True/False masks as pandas data series, which can be used as masks for
    the "matches" pandas dataframe which stores match information for each
    edge of the graph.

    Attributes
    ----------

    """
    def __init__(self):
        pass

    # (query only takes care of literal self-matches on a keypoint basis, not self-matches for the whole image)
    def self_neighbors(self, matches):
        """
        Returns a pandas data series intended to be used as a mask. Each row
        is True if it is not matched to a point in the same image (good) and
        False if it is (bad.)

        Parameters
        ----------
        matches : dataframe
                  the matches dataframe stored along the edge of the graph
                  containing matched points with columns containing:
                  matched image name, query index, train index, and
                  descriptor distance
        Returns
        -------
        : dataseries
          Intended to mask the matches dataframe. True means the row is not matched to a point in the same image
          and false the row is.
        """
        return matches.source_image != matches.destination_image

    def distance_ratio(self, matches, ratio=0.8):
        """
        Compute and return a mask for the matches dataframe stored on each edge of the graph
        using the ratio test and distance_ratio set during initialization.

        Parameters
        ----------
        matches : dataframe
                  the matches dataframe stored along the edge of the graph
                  containing matched points with columns containing:
                  matched image name, query index, train index, and
                  descriptor distance. ***Will only work as expected if matches already has dropped duplicates***

        ratio: float
               the ratio between the first and second-best match distances
               for each keypoint to use as a bound for marking the first keypoint
               as "good."
        Returns
        -------
         : dataseries
           Intended to mask the matches dataframe. Rows are True if the associated keypoint passes
           the ratio test and false otherwise. Keypoints without more than one match are True by
           default, since the ratio test will not work for them.
        """
        #0.8 is Lowe's paper value -- can be changed.
        mask = []
        temp_matches = matches.drop_duplicates() #don't want to deal with duplicates...
        for key, group in temp_matches.groupby('source_idx'):
            #won't work if there's only 1 match for each queryIdx
            if len(group) < 2:
                mask.append(True)
            else:
                if group['distance'].iloc[0] < ratio * group['distance'].iloc[1]: #this means distance _0_ is good and can drop all other distances
                    mask.append(True)
                    for i in range(len(group['distance']-1)):
                        mask.append(False)
                else:
                    for i in range(len(group['distance'])):
                        mask.append(False)
        return pd.Series(mask)

    def mirroring_test(self, matches):
        """
        Compute and return a mask for the matches dataframe on each edge of the graph which
        will keep only entries in which there is both a source -> destination match and a destination ->
        source match.

        Parameters
        ----------
        matches : dataframe
                  the matches dataframe stored along the edge of the graph
                  containing matched points with columns containing:
                  matched image name, query index, train index, and
                  descriptor distance

        Returns
        -------
         : dataseries
           Intended to mask the matches dataframe. Rows are True if the associated keypoint passes
           the mirroring test and false otherwise. That is, if 1->2, 2->1, both rows will be True,
           otherwise, they will be false. Keypoints with only one match will be False. Removes
           duplicate rows.
        """
        return matches.duplicated(keep='first')




