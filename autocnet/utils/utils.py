from functools import reduce
import numpy as np
import pandas as pd
import math

import warnings

def normalize_vector(line):
    """
    Normalize a standard form line

    Parameters
    ----------
    line : ndarray
           Standard form of a line (Ax + By + C = 0)

    Returns
    -------
    line : ndarray
           The normalized line

    Examples
    --------
    >>> x = np.random.random((3,3))
    >>> normalize_vector(x)
    array([[ 0.88280225,  0.4697448 ,  0.11460811],
       [ 0.26090555,  0.96536433,  0.91648305],
       [ 0.58271501,  0.81267657,  0.30796395]])
    """
    if isinstance(line, pd.DataFrame):
        line = line.values
    try:
        n = np.sqrt(line[:, 0]**2 + line[:, 1]**2).reshape(-1, 1)
    except:
        n = np.sqrt(line[0]**2 + line[1]**2)
    line = line / n
    return line


def getnearest(iterable, value):
    """
    Given an iterable, get the index nearest to the input value

    Parameters
    ----------
    iterable : iterable
               An iterable to search

    value : int, float
            The value to search for

    Returns
    -------
        : int
          The index into the list
    """
    return min(enumerate(iterable), key=lambda i: abs(i[1] - value))


def checkbandnumbers(bands, checkbands):
    """
    Given a list of input bands, check that the passed
    tuple contains those bands.

    In case of THEMIS, we check for band 9 as band 9 is the temperature
    band required to derive thermal temperature.  We also check for band 10
    which is required for TES atmosphere calculations.

    Parameters
    ----------
    bands : tuple
            of bands in the input image
    checkbands : list
                 of bands to check against

    Returns
    -------
     : bool
       True if the bands are present, else False
    """
    for c in checkbands:
        if c not in bands:
            return False
    return True


def checkdeplaid(incidence):
    """
    Given an incidence angle, select the appropriate deplaid method.

    Parameters
    ----------
    incidence : float
                incidence angle extracted from the campt results.

    """
    if incidence >= 95 and incidence <= 180:
        return 'night'
    elif incidence >=90 and incidence < 95:
        return 'night'
    elif incidence >= 85 and incidence < 90:
        return 'day'
    elif incidence >= 0 and incidence < 85:
        return 'day'
    else:
        return False


def checkmonotonic(iterable, piecewise=False):
    """
    Check if a given iterable is monotonically increasing.

    Parameters
    ----------
    iterable : iterable
                Any Python iterable object

    piecewise : boolean
                If false, return a boolean for the entire iterable,
                else return a list with elementwise monotinicy checks

    Returns
    -------
    monotonic : bool/list
                A boolean list of all True if monotonic, or including
                an inflection point
    """
    monotonic = [True] + [x < y for x, y in zip(iterable, iterable[1:])]
    if piecewise is True:
        return monotonic
    else:
        return all(monotonic)


def find_in_dict(obj, key):
    """
    Recursively find an entry in a dictionary

    Parameters
    ----------
    obj : dict
          The dictionary to search
    key : str
          The key to find in the dictionary

    Returns
    -------
    item : obj
           The value from the dictionary
    """
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            item = find_in_dict(v, key)
            if item is not None:
                return item


def find_nested_in_dict(data, key_list):
    """
    Traverse a list of keys into a dict.

    Parameters
    ----------
    data : dict
           The dictionary to be traversed
    key_list: list
              The list of keys to be travered.  Keys are
              traversed in the order they are entered in
              the list

    Returns
    -------
    value : object
            The value in the dict
    """
    return reduce(lambda d, k: d[k], key_list, data)


def make_homogeneous(points):
    """
    Convert a set of points (n x dim array) to
        homogeneous coordinates.

    Parameters
    ----------
    points : ndarray
             n x m array of points, where n is the number
             of points.

    Returns
    -------
     : ndarray
       n x m + 1 array of homogeneous points
    """
    return np.hstack((points, np.ones((points.shape[0], 1))))


def remove_field_name(a, name):
    """
    Given a numpy structured array, remove a column and return
    a copy of the remainder of the array

    Parameters
    ----------
    a : ndarray
        Numpy structured array

    name : str
           of the index (column) to be removed

    Returns
    -------
    b : ndarray
        Numpy structured array with the 'name' column removed
    """
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]
    return b


def cartesian_product(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian_product(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def corr_normed(template, search):
    """
    Numpy implementation of normalized cross-correlation

    parameters
    ----------
    template : ndarray
               template to use for correlation

    search : ndarray
             model to correlate the template to

    returns
    -------

    result : float
             correlation strength
    """

    if template.shape != search.shape:
        warnings.warn('Input image are not of the same size, truncating to common values.'
                      'for template size: {} and search size {}'.format(template.shape, search.shape))
        if len(template) < len(search):
            search = search[:len(template)]
        elif len(template) > len(search):
            template = template[:len(search)]

    # get averages
    template_avg = np.average(template)
    search_avg = np.average(search)

    # compute mean-corrected vetcors
    mc_template = [x-template_avg for x in template]
    mc_search = [x-search_avg for x in search]

    # Perform element-wise multiplication
    arr1xarr2 = np.multiply(mc_template, mc_search)

    # element-wise divide by the mangitude1 x magnitude2
    # and return the result
    std1xstd2 = np.std(template) * np.std(search)

    coeffs = [(x/std1xstd2) for x in arr1xarr2]
    return np.average(coeffs)


def to_polar_coord(shape, center):
    """
    Generate a polar coordinate grid from a shape given
    a center.

    parameters
    ----------
    shape : tuple
            tuple decribing the desired shape in
            (y,x)

    center : tuple
             tuple describing the desired center
             for the grid

    returns
    -------
    r2 : ndarray
         grid of radii from the center

    theta : ndarray
            grid of angles from the center
    """

    y, x = np.ogrid[:shape[0], :shape[1]]
    cy, cx = center
    tmin, tmax = (0, 2*math.pi)

    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    return r2, theta


def circ_mask(shape, center, radius):
    """
    Generates a circular mask

    parameters
    ----------
    shape : tuple
            tuple decribing the desired mask shape in
            (y,x)

    center : tuple
             tuple describing the desired center
             for the circle

    radius : float
             radius of circlular mask

    returns
    -------

    mask : ndarray
           circular mask of bools
    """

    r, theta = to_polar_coord(shape, center)

    circle_mask = r == radius*radius
    angle_mask = theta <= 2*math.pi

    return circle_mask*angle_mask


def radial_line_mask(shape, center, radius, alpha=0.19460421, atol=.01):
    """
    Generates a linear mask from center at angle alpha.

    parameters
    ----------
    shape : tuple
            tuple decribing the desired mask shape in
            (y,x)

    center : tuple
             tuple describing the desired center
             for the circle

    radius : float
             radius of the line mask

    alpha : float
            angle for the line mask

    atol : float
           absolute tolerance for alpha, the higher
           the tolerance, the wider the angle bandwidth

    returns
    -------

    mask : ndarray
           linear mask of bools
    """

    r, theta = to_polar_coord(shape, center)

    line_mask = r <= radius**2
    anglemask = np.isclose(theta, alpha, atol=atol)

    return line_mask*anglemask
