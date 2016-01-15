import cv2
from scipy import misc

def extract_features(image_array, extractor_parameters):
    """
    This method finds and extracts features from an image using the given dictionary of keyword arguments. 
    The input image is represented as NumPy array and the output features are represented as keypoint IDs 
    with corresponding descriptors.

    Parameters
    ----------
    image_array : ndarray
                  a NumPy array that represents an image
    extractor_parameters : dict
                           A dictionary containing OpenCV SIFT parameters names and values. 

    Returns
    -------
    : tuple
      in the form ([list of OpenCV KeyPoints], [NumPy array of descriptors as geometric vectors])
    """
    sift = cv2.xfeatures2d.SIFT_create(**extractor_parameters)
    converted_array = misc.bytescale(image_array)
    return sift.detectAndCompute(converted_array, None)
