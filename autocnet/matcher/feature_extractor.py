import cv2

from scipy.misc import bytescale

try:
    import cyvlfeat as vl
    vlfeat = True
except:
    vlfeat = False
    pass


def extract_features(array, method='orb', extractor_parameters={}):
    """
    This method finds and extracts features from an image using the given dictionary of keyword arguments.
    The input image is represented as NumPy array and the output features are represented as keypoint IDs
    with corresponding descriptors.

    Parameters
    ----------
    array : ndarray
            a NumPy array that represents an image

    method : {'orb', 'sift', 'fast', 'surf', 'vl_sift'}
              The detector method to be used.  Note that vl_sift requires that
              vlfeat and cyvlfeat dependencies be installed.

    extractor_parameters : dict
                           A dictionary containing OpenCV SIFT parameters names and values.

    Returns
    -------
    : tuple
      in the form ([list of OpenCV KeyPoints], [NumPy array of descriptors as geometric vectors])
    """

    detectors = {'fast': cv2.FastFeatureDetector_create,
                 'sift': cv2.xfeatures2d.SIFT_create,
                 'surf': cv2.xfeatures2d.SURF_create,
                 'orb': cv2.ORB_create}
    if vlfeat:
        detectors['vl_sift'] = vl.sift.sift

    if 'vl_' in method:
        return detectors[method](array, compute_descriptor=True, float_descriptors=True, **extractor_parameters)
    else:
        # OpenCV requires the input images to be 8-bit
        if not array.dtype == 'int8':
            array = bytescale(array)
        detector = detectors[method](**extractor_parameters)
        return detector.detectAndCompute(array, None)
