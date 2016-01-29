import pandas as pd
from autocnet.matcher import matcher

from scipy.misc import imresize

# TODO: look into KeyPoint.size and perhaps use to determine an appropriately-sized search/template.
# TODO: do not allow even sizes

def subpixel_offset(template_kp, search_kp, template_img, search_img, template_size=9, search_size=27, upsampling=10):
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
    # Get the x,y coordinates
    temp_x, temp_y = map(int, template_kp.pt)
    search_x, search_y = map(int, search_kp.pt)

    # Convert desired template and search sizes to offsets to get the bounding box
    t = int(template_size/2) #index offset for template
    s = int(search_size/2) #index offset for search

    template = template_img[temp_y-t:temp_y+t, temp_x-t:temp_x+t]
    search = search_img[search_y-s:search_y+s, search_x-s:search_x+s]

    results = (None, None, None)

    try:
        results = matcher.pattern_match(template, search, upsampling=upsampling)
    except ValueError:
        # the match fails if the template or search point is near an edge of the image
        # TODO: come up with a better solution?
        print('Template Keypoint ({},{}) cannot be pattern matched'.format(str(temp_x), str(temp_y)))

    return results
