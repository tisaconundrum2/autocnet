import numpy as np
import pandas as pd


def compute_epipoles(f):
    """
    Compute the epipole and epipolar prime

    Parameters
    ----------
    f : ndarray
        (3,3) fundamental matrix or autocnet Fundamental Matrix object

    Returns
    -------
    e : ndarray
        (3,1) epipole

    e1 : ndarray
         (3,3) epipolar prime matrix
    """
    u, _, _ = np.linalg.svd(f)
    e = u[:, -1]
    e1 = np.array([[0, -e[2], e[1]],
                   [e[2], 0, -e[0]],
                   [-e[1], e[0], 0]])

    return e, e1


def idealized_camera():
    """
    Create an idealized camera transformation matrix

    Returns
    -------
     : ndarray
       (3,4) with diagonal 1
    """
    return np.eye(3, 4)


def estimated_camera_from_f(f):
    """
    Estimate a camera matrix using a fundamental matrix.


    Parameters
    ----------
    f : ndarray
        (3,3) fundamental matrix or autocnet Fundamental Matrix object

    Returns
    -------
    p1 : ndarray
         Estimated camera matrix
    """

    e, e1 = compute_epipoles(f)
    p1 = np.empty((3, 4))
    p1[:, :3] = e1.dot(f)
    p1[:, 3] = e

    return p1


def triangulate(pt, pt1, p, p1):
    """
    Given two sets of homogeneous coordinates and two camera matrices,
    triangulate the 3D coordinates.  The image correspondences are
    assumed to be implicitly ordered.

    References
    ----------
    .. [Hartley2003]

    Parameters
    ----------
    pt : ndarray
         (n, 3) array of homogeneous correspondences

    pt1 : ndarray
          (n, 3) array of homogeneous correspondences

    p : ndarray
        (3, 4) camera matrix

    p1 : ndarray
         (3, 4) camera matrix

    Returns
    -------
    coords : ndarray
             (n, 4) array of triangulated coordinates in the form (x, y, z, a)

    """
    npts = len(pt)
    a = np.zeros((4, 4))
    coords = np.empty((npts, 4))

    if isinstance(pt, pd.DataFrame):
        pt = pt.values
    if isinstance(pt1, pd.DataFrame):
        pt1 = pt.values

    for i in range(npts):
        # Compute AX = 0
        a[0] = pt[i][0] * p[2] - p[0]
        a[1] = pt[i][1] * p[2] - p[1]
        a[2] = pt1[i][0] * p1[2] - p1[0]
        a[3] = pt1[i][1] * p1[2] - p1[1]
        # v.T is a least squares solution that minimizes the error residual
        u, s, vh = np.linalg.svd(a)
        v = vh.T
        coords[i] = v[:, 3] / (v[:, 3][-1])
    return coords.T


def projection_error(p1, p, pt, pt1):
    """
    Based on Hartley and Zisserman p.285 this function triangulates
    image correspondences and computes the reprojection error
    by back-projecting the points into the image.

    References
    ----------
    .. [Hartley2003]

    Parameters
    -----------
    p1 : ndarray
         (3,4) camera matrix

    p : ndarray
        (3,4) idealized camera matrix in the form np.eye(3,4)

    pt : dataframe or ndarray
         of homogeneous coordinates in the form (x_{i}, y_{i}, 1)

    pt1 : dataframe or ndarray
          of homogeneous coordinates in the form (x_{i}, y_{i}, 1)

    Returns
    -------
    reproj_error : ndarray
                   (n, 1) residuals for each correspondence

    """
    if p1.shape != (3,4):
        p1 = p1.reshape(3,4)
    # Triangulate the correspondences
    xw_est = triangulate(pt, pt1, p, p1)

    xhat = p.dot(xw_est).T
    xhat /= xhat[:, -1][:, np.newaxis]
    x2hat = p1.dot(xw_est).T
    x2hat /= x2hat[:, -1][:, np.newaxis]

    # Compute residuals
    dist = (pt - xhat)**2 + (pt1 - x2hat)**2
    reproj_error = np.sqrt(np.sum(dist, axis=1) / len(pt))

    return reproj_error
