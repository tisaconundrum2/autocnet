import numpy as np

from autocnet.matcher import matcher

# TODO: look into KeyPoint.size and perhaps use to determine an appropriately-sized search/template.


def clip_roi(img, center, img_size):
    """
    Given an input image, clip a square region of interest
    centered on some pixel at some size.

    Parameters
    ----------
    img : ndarray or file handle
          The input image to be clipped

    center : tuple
             (y,x) coordinates to center the roi

    img_size : int
               Odd, total image size

    Returns
    -------
    clipped_img : ndarray
                  The clipped image
    """
    if img_size % 2 == 0:
            raise ValueError('Image size must be odd.')

    i = (img_size - 1) / 2

    y, x = map(int, center)

    if isinstance(img, np.ndarray):
        clipped_img = img[y - i:y + i,
                          x - i:x + i]

    return clipped_img


def subpixel_offset(template, search, upsampling=10):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.

    Parameters
    ----------
    template_kp : KeyPoint
                  The KeyPoint to match the search_kp to.
    search_kp : KeyPoint
                The KeyPoint to match to the template_kp
    template_img : numpy array
                   The entire image that the template chip to match to will be taken out of.
    search_img : numpy array
                 The entire image that the search chip to match to the template chip will be taken out of.
    template_size : int
                    The length of one side of the square subset of the template image that will actually be used for
                    the subpixel registration. Default is 9.
                    Must be odd.
    search_size : int
                  The length of one side of the square subset of the search image that will be used for subpixel
                  registration. Default is 13. Must be odd.
    Returns
    -------
    : tuple
      The returned tuple is of form: (x_offset, y_offset, strength). The offsets are from the search to the template
      keypoint.
    """

    try:
        results = matcher.pattern_match(template, search, upsampling=upsampling)
        return results
    except ValueError:
        # the match fails if the template or search point is near an edge of the image
        # TODO: come up with a better solution?
        print('Can not subpixel match point.')
        return
