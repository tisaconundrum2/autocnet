import warnings

import cv2
import pandas as pd
import numpy as np
import math

from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.interpolation import rotate

import autocnet.utils.utils as util

FLANN_INDEX_KDTREE = 1  # Algorithm to set centers,
DEFAULT_FLANN_PARAMETERS = dict(algorithm=FLANN_INDEX_KDTREE,
                                trees=3)


def pattern_match(template, image, upsampling=16, func=cv2.TM_CCOEFF_NORMED, error_check=False):
    """
    Call an arbitrary pattern matcher

    Parameters
    ----------
    template : ndarray
               The input search template used to 'query' the destination
               image

    image : ndarray
            The image or sub-image to be searched

    upsampling : int
                 The multiplier to upsample the template and image.

    func : object
           The function to be used to perform the template based matching
           Options: {cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED, cv2.TM_SQDIFF_NORMED}
           In testing the first two options perform significantly better with Apollo data.

    error_check : bool
                  If True, also apply a different matcher and test that the values
                  are not too divergent.  Default, False.

    Returns
    -------

    x : float
        The x offset

    y : float
        The y offset

    strength : float
               The strength of the correlation in the range [-1, 1].
    """

    different = {cv2.TM_SQDIFF_NORMED: cv2.TM_CCOEFF_NORMED,
                 cv2.TM_CCORR_NORMED: cv2.TM_SQDIFF_NORMED,
                 cv2.TM_CCOEFF_NORMED: cv2.TM_SQDIFF_NORMED}

    if upsampling < 1:
        raise ValueError

    u_template = zoom(template, upsampling, order=1)
    u_image = zoom(image, upsampling, order=1)

    result = cv2.matchTemplate(u_image, u_template, method=func)
    min_corr, max_corr, min_loc, max_loc = cv2.minMaxLoc(result)
    if func == cv2.TM_SQDIFF or func == cv2.TM_SQDIFF_NORMED:
        x, y = (min_loc[0], min_loc[1])
    else:
        x, y = (max_loc[0], max_loc[1])

    # Compute the idealized shift (image center)
    ideal_y = u_image.shape[0] / 2
    ideal_x = u_image.shape[1] / 2

    # Compute the shift from template upper left to template center
    y += (u_template.shape[0] / 2)
    x += (u_template.shape[1] / 2)

    x = (ideal_x - x) / upsampling
    y = (ideal_y - y) / upsampling
    return x, y, max_corr


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
        self.nid_lookup = {}
        self.node_counter = 0

    def add(self, descriptor, nid):
        """
        Add a set of descriptors to the matcher and add the image
        index key to the image_indices attribute

        Parameters
        ----------
        descriptor : ndarray
                     The descriptor to be added

        nid : int
              The node ids
        """
        self._flann_matcher.add([descriptor])
        self.nid_lookup[self.node_counter] = nid
        self.node_counter += 1

    def clear(self):
        """
        Remove all nodes from the tree and resets
        all counters
        """
        self._flann_matcher.clear()
        self.nid_lookup = {}
        self.node_counter = 0

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
                source = query_image
                destination = self.nid_lookup[i.imgIdx]
                if source < destination:
                    matched.append((query_image,
                                    i.queryIdx,
                                    destination,
                                    i.trainIdx,
                                    i.distance))
                elif source > destination:
                    matched.append((destination,
                                    i.trainIdx,
                                    query_image,
                                    i.queryIdx,
                                    i.distance))
                else:
                    warnings.warn('Likely self neighbor in query!')
        return pd.DataFrame(matched, columns=['source_image', 'source_idx',
                                              'destination_image', 'destination_idx',
                                              'distance'])


def cifi(template, search_image, thresh=95, use_percentile=True,
         radii=list(range(1,12)), scales=[0.5, 0.57, 0.66,  0.76, 0.87, 1.0]):
    """
        Circular sampling filter (Cifi) uses projections of the template and search
        images on a set of circular rings to detect the first grade candidate pixels
        and the points' corresponding best fit scales for Ciratefi.

        A set of scales is applied to the template and is radially sampled for each radii
        'r' passed in. The template sample is equal to sum of the grayscale values
        divided by 2*pi*r.

        Each pixel in the search image is similarly sampled. Every pixel then gets
        correlated with the template samples at all scales. The scales with the highest
        correlation are the considered the 'best fit scales'.

        parameters
        ----------
        template : ndarray
                   The input search template used to 'query' the destination
                   image

        image : ndarray
                The image or sub-image to be searched

        thresh : float
                 The correlation thresh hold above which a point will
                 be a first grade candidate point. If use_percentile=True
                 this will act as a percentile, for example, passing 90 means
                 keep values in the top 90th percentile

        use_percentile : bool
                         If True (default), thresh is a percentile instead of a hard
                         strength value

        radii : list
                The list of radii to use for radial sampling

        scales : list
                 The list of scales to be applied to the template image, best if
                 a geometric series

        returns
        -------
        fg_candidate_pixels : 1darray
                              array of pixels that passed the filter in tuples (y,x)

        best_scales : 1darray
                      parrallel array of best scales for the first grade candidate points
    """

    if template.shape > search_image.shape:
        raise ValueError('Template Image is smaller than Search Image for template of\
                          size: {} and search image of size: {}'
                         .format(template.shape, search_image.shape))

    # Cifi -- Circular Sample on Template
    template_result = np.empty((len(scales), len(radii)))

    for i, s in enumerate(scales):
        scaled_img = imresize(template, s)
        for j, r in enumerate(radii):
            # Generate a circular mask
            a, b = (int(scaled_img.shape[0] / 2),
                    int(scaled_img.shape[1] / 2))

            if r > b or r > a:
                s =-1

            mask = util.circ_mask(scaled_img.shape, (a,b), r)

            inv_area = 1 / (2 * math.pi * r)
            s = np.sum(scaled_img[mask]) * inv_area
            if s == 0:
                s = -1
            template_result[i,j] = s

    # Cifi2 -- Circular Sample on Target Image
    search_result = np.empty((search_image.shape[0], search_image.shape[1], len(radii)))

    for i, y in enumerate(range(search_image.shape[0])):
        for j, x in enumerate(range(search_image.shape[1])):
            for k, r in enumerate(radii):
                inv_area = 1 / (2 * math.pi * r)

                mask = util.circ_mask(search_image.shape, (i,j), r)
                s = np.sum(search_image[mask]) * inv_area

                if s == 0 or y<r or x<r or y+r>search_image.shape[0] or x+r>search_image.shape[1]:
                    s = -1
                search_result[i, j, k] = s

    # Perform Normalized Cross-Correlation between template and target image
    coeffs = np.empty((search_result.shape[0], search_result.shape[1]))
    best_scales = np.empty((search_result.shape[0], search_result.shape[1]))

    for y in range(search_result.shape[0]):
        for x in range(search_result.shape[1]):
            scale = 0
            max_coeff = -math.inf
            for i in range(template_result.shape[0]):

                max_corr = util.corr_normed(template_result[i], search_result[y,x])

                if max_corr > max_coeff:
                    max_coeff = max_corr
                    scale = i

            coeffs[y,x] = max_coeff
            best_scales[y,x] = scales[scale]

    a, b = (int(search_image.shape[0] / 2),
        int(search_image.shape[1] / 2))

    # get first grade candidate points
    t1 = thresh if not use_percentile else np.percentile(coeffs, thresh)
    fg_candidate_pixels = np.array([(y, x) for (y, x), coeff in np.ndenumerate(coeffs) if coeff >= t1])

    if fg_candidate_pixels.size == 0:
        raise warnings.warn('Cifi returned empty set.')

    return fg_candidate_pixels, best_scales


def rafi(template, search_image, candidate_pixels, scales, thresh=95,
         use_percentile=True, alpha=math.pi/16, radii=list(range(1,12))):
    """
    The seconds filter in Ciratefi, the Radial Sampling Filter (Rafi), uses
    projections of the template image and the search image on a set of radial
    lines to upgrade the first grade the candidate pixels from cefi to
    seconds grade candidate pixels along with there corresponding best
    fit rotation.

    The template image is radially sampled at angles 0-2*pi at steps alpha and
    with the best fit radius (largest sampling radius from radii list that fits
    in the template image)

    Sampling for each line equals the sum of the greyscales divided by the
    best fit radius.

    The search image is similarly sampled at each candidate pixel and is correlated
    with the radial samples on the template. The best fit angle is the angle that
    maximizes this correlation, and the second grade candidate pixels are determined
    by the strength of the correlation and the passed threshold

    parameters
    ----------
    template : ndarray
               The input search template used to 'query' the destination
               image

    image : ndarray
            The image or sub-image to be searched

    candidate_pixels : 1darray
                       array of candidate pixels in tuples (y,x), best if
                       the pixel are the output of Cifi

    scales : list
             The list of best fit scales for each candidate point,
             the length should equal the length of the candidate point
             list

    thresh : float
             The correlation thresh hold above which a point will
             be a first grade candidate point. If use_percentile=True
             this will act as a percentile, for example, passing 90 means
             keep values in the top 90th percentile

    use_percentile : bool
                     If True (default), thresh is a percentile instead of a hard
                     strength value

    alpha : float
            A float between 0 & 2*pi, alpha list = np.arange(0, 2*pi, alpha)

    radii : list
            The list of radii to use for radial sampling, best if the list
            is the same as the one used for Cifi

    returns
    -------
    sg_candidate_points : 1darray

    best_rotation : 1darray
                    Parrallel array of the best fit rotations for each
                    second grade candidate pixel
    """

    # Rafi 1  -- Get Radial Samples of Template Image
    alpha_list = np.arange(0, 2*math.pi, alpha)
    template_alpha_samples = np.zeros(len(alpha_list))
    center_y, center_x = (int(template.shape[0] / 2),
                          int(template.shape[1] / 2))

    # find the largest fitting radius
    rad_thresh = center_x if center_x <= center_y else center_y

    if rad_thresh >= max(radii):
        radius = max(radii)
    else:
        radius = radii[np.bisect_left(radii, rad_thresh)]

    for i in range(len(template_alpha_samples)):
        # Create Radial Line Mask
        mask = util.radial_line_mask(template.shape, (center_y, center_x), radius, alpha=alpha_list[i])

        # Sum the values
        template_alpha_samples[i] = np.sum(template[mask])/radius

    # Rafi 2 -- Get Radial Samples of the Search Image for all First Grade Candidate Points
    rafi_alpha_means = np.zeros((len(candidate_pixels), len(alpha_list)))

    for i in range(len(candidate_pixels)):
        y, x = candidate_pixels[i]

        rad = radius if min(y,x) > radius else min(y,x)
        cropped_search = search_image[y-rad:y+rad+1, x-rad:x+rad+1]
        scaled_img = imresize(cropped_search, scales[y,x])

        # Will except if image size too small after scaling
        try:
            scaled_center_y, scaled_center_x = (math.floor(scaled_img.shape[0]/2),
                                                math.floor(scaled_img.shape[1]/2))
        except:
            rafi_alpha_means[i] = np.negative(np.ones(len(alpha_list)))
            continue

        for j in range(len(alpha_list)):
            # Create Radial Mask
            mask = util.radial_line_mask(scaled_img.shape, (scaled_center_y, scaled_center_x),
                                         scaled_center_y, alpha=alpha_list[j])
            rafi_alpha_means[i,j] = np.sum(scaled_img[mask])/radius

    coeffs = np.zeros((search_image.shape[0], search_image.shape[1]))
    best_rotation = np.zeros(len(candidate_pixels))
    rafi_coeffs = np.zeros(len(candidate_pixels))

    # Perform Normalized Cross-Correlation between template and target image
    for i in range(len(candidate_pixels)):
        maxcoeff = -math.inf
        maxrotate = 0
        y, x = candidate_pixels[i]
        for j in range(len(alpha_list)):
            c_shift_RQ = np.roll(template_alpha_samples, j)
            score = util.corr_normed(c_shift_RQ, rafi_alpha_means[i])

            if score > maxcoeff:
                maxcoeff = score
                maxrotate = j

        coeffs[y,x] = maxcoeff
        rafi_coeffs[i] = maxcoeff
        best_rotation[i] = alpha_list[maxrotate]

    # Get second grade candidate points and best rotation
    t2 = thresh if not use_percentile else np.percentile(rafi_coeffs, thresh)
    rafi_mask = rafi_coeffs >= t2
    sg_candidate_points = candidate_pixels[rafi_mask]
    best_rotation = best_rotation[rafi_mask]

    if(sg_candidate_points.size == 0):
        warnings.warn('Second filter Rafi returned empty set.')

    return sg_candidate_points, best_rotation

def tefi(template, search_image, candidate_pixels, best_scales, best_angles,
         scales=[0.5, 0.57, 0.66,  0.76, 0.87, 1.0], upsampling=1, thresh=100,
         alpha=math.pi/16, use_percentile=True):
    """
    Template Matching Filter (Tefi) is the third and final filter for ciratefi.

    For every candidate pixel, tefi rotates and scales the template image by the list
    of scales and angles passed in (which, ideally are the output from cefi and rafi
    respectively) and performs template match around the candidate pixels at the
    approriate scale and rotation angle. Here, the scales, angles and candidate
    points should be a parrallel array structure.

    Any points with correlation strength over the threshold are returned as
    the the strongest candidates for the image location. If knows the point
    exists in one location, thresh should be 100 and use_percentile = True.

    parameters
    ----------
    template : ndarray
               The input search template used to 'query' the destination
               image

    image : ndarray
            The image or sub-image to be searched

    candidate_pixels : 1darray
                       array of candidate pixels in tuples (y,x), best if
                       the pixel are the output of Cifi

    best_scales : list
                  The list of best fit scales for each candidate point, the length
                  should equal the length of the candidate point list

    best_angles : list
                  The list of best fit rotation for each candidate point in radians,
                  the length should equal the length of the candidate point list

    upsampling : int
                 upsample degree

    thresh : float
             The correlation thresh hold above which a point will
             be a first grade candidate point. If use_percentile=True
             this will act as a percentile, for example, passing 90 means
             keep values in the top 90th percentile

    use_percentile : bool
                     If True (default), thresh is a percentile instead of a hard
                     strength value

    alpha : float
            A float between 0 & 2*pi, alpha list = np.arange(0, 2*pi, alpha)


    returns
    -------

    results : 1darray
              array of pixel tuples (y,x) which over the threshold
    """

    # Check inputs
    if upsampling < 1:
        raise ValueError('Upsampling must be >= 1, got {}'.format(upsampling))

    coeffs = np.zeros((search_image.shape[0], search_image.shape[1]))
    tefi_coeffs = np.zeros(len(candidate_pixels))

    # check for upsampling
    if upsampling > 1:
        template = zoom(template, upsampling, order=3)
        search_image = zoom(search_image, upsampling, order=3)

    alpha_list = np.arange(0, 2*math.pi, alpha)
    candidate_pixels *= int(upsampling)

    # Tefi -- Template Matching Filter
    for i in range(len(candidate_pixels)):
        y, x = candidate_pixels[i]

        best_scale_idx = (np.where(scales == best_scales[y//upsampling, x//upsampling]))[0][0]
        best_alpha_idx = (np.where(alpha_list == best_angles[i]))[0][0]

        tefi_scales = np.array(scales).take(range(best_scale_idx-1, best_scale_idx+2), mode='wrap')
        tefi_alphas = alpha_list.take(range(best_alpha_idx-1, best_alpha_idx+2), mode='wrap')

        scalesXalphas = util.cartesian_product([tefi_scales, tefi_alphas])

        max_coeff = -math.inf
        for j in range(scalesXalphas.shape[0]):
            transformed_template = imresize(template, scalesXalphas[j][0])
            transformed_template = rotate(transformed_template, scalesXalphas[j][1])

            y_window, x_window = (math.floor(transformed_template.shape[0]/2),
                                  math.floor(transformed_template.shape[1]/2))

            cropped_search = search_image[y-y_window:y+y_window+1, x-x_window:x+x_window+1]

            if y < y_window or x < x_window or cropped_search.shape < transformed_template.shape:
                score = -1
            else:
                score = util.corr_normed(transformed_template.flatten(), cropped_search.flatten())

            if score > max_coeff:
                max_coeff = score

        coeffs[y//upsampling][x//upsampling] = max_coeff
        tefi_coeffs[i] = max_coeff

    t3 = thresh if not use_percentile else np.percentile(tefi_coeffs, thresh)
    results = candidate_pixels[np.where(tefi_coeffs >= t3)]

    return  results//upsampling


def ciratefi(template, search_image, upsampling=1, cifi_thresh=95, rafi_thresh=95, tefi_thresh=100,
             use_percentile=False, alpha=math.pi/16,
             scales=[0.5, 0.57, 0.66,  0.76, 0.87, 1.0], radii=list(range(1,12))):

    # Perform first filter cifi
    fg_candidate_pixels, best_scales = cifi(template, search_image, thresh=cifi_thresh,
                                            use_percentile=use_percentile,
                                            scales=scales, radii=radii)
    # Perform second filter rafi
    sg_candidate_points, best_rotation = rafi(template, search_image, fg_candidate_pixels, best_scales,
                                              thresh = rafi_thresh, use_percentile=use_percentile, alpha=alpha,
                                              radii=radii)

    # Perform last filter tefi
    results = tefi(template, search_image, sg_candidate_points, best_scales, best_rotation,
                   thresh=tefi_thresh, alpha=math.pi/4, use_percentile=True, upsampling=upsampling)

    # return the points found
    return results

















