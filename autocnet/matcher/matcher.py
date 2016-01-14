import cv2
import numpy as np

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
        self.image_index_counter = 0  # OpenCv DMatch.imgIdx are one based

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
         :
        """
        print(self.image_indices)
        matches = self._flann_matcher.knnMatch(descriptor, k=k)
        print(dir(self._flann_matcher))
        print(matches)
        for m in matches:
            print(m[0].imgIdx, m[0].queryIdx, m[0].trainIdx)
            #print(self.image_indices[m[0].imgIdx])


def match_features(iterable, flann_parameters=DEFAULT_FLANN_PARAMETERS):
    """
    Iterate over all nodes in the graph, create a single KDTree,
    and then apply a FLANN matcher.

    Parameters
    ----------
    iterable : iterable
            An iterable of descriptors

    flann_parameters : dict
                       of parameters for the FLANN matcher


    Returns
    -------

    """
    flann_matcher = create_flann_matcher(iterable, flann_parameters)


def create_flann_matcher(iterable, flann_parameters):

    flann_matcher = cv2.FlannBasedMatcher(flann_parameters, {})

    try:
        if isinstance(iterable, CandidateGraph):
            training_descriptors = [i[1]['descriptors'] for i in iterable.nodes_iter(data=True)]
        elif isinstance(iterable[0], np.ndarray):
            training_descriptors = iterable
        else:
            raise(TypeError, 'Unsupported iterable passed to match_features')
    except TypeError:
        print('Object is not iterable')

    flann_matcher.add(training_descriptors)
    flann_matcher.train()
    return flann_matcher