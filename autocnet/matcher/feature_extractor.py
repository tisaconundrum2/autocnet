import cv2


def extract_features(array, extractor_parameters):
    """
    This method finds and extracts features from an image using the given dictionary of keyword arguments. 
    The input image is represented as NumPy array and the output features are represented as keypoint IDs 
    with corresponding descriptors.

    Parameters
    ----------
    array : ndarray
            a NumPy array that represents an image

    extractor_parameters : dict
                           A dictionary containing OpenCV SIFT parameters names and values. 

    Returns
    -------
    : tuple
      in the form ([list of OpenCV KeyPoints], [NumPy array of descriptors as geometric vectors])
    """

    sift = cv2.xfeatures2d.SIFT_create(**extractor_parameters)
    return sift.detectAndCompute(array, None)
