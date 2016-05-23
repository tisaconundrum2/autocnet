import json
import ogr
import pandas as pd

from autocnet.fileio import io_gdal
from scipy.spatial import ConvexHull


def convex_hull_ratio(points, ideal_area):
    """

    Parameters
    ----------
    points : ndarray
             (n, 2) array of point coordinates

    ideal_area : float
                 The total area that could be covered

    Returns
    -------
    ratio : float
            The ratio convex hull volume / ideal_area

    """
    hull = ConvexHull(points)
    return hull.volume / ideal_area


def overlapping_polygon_area(polys):
    """

    Parameters
    ----------
    polys : list
            of polygon object with a __geo_interface__

    Returns
    -------
    area : float
           The area of the intersecting polygons
    """
    geom = json.dumps(polys[0].__geo_interface__)
    intersection = ogr.CreateGeometryFromJson(geom)
    for p in polys[1:]:
        geom = json.dumps(p.__geo_interface__)
        geom = ogr.CreateGeometryFromJson(geom)
        intersection = intersection.Intersection(geom)
    area = intersection.GetArea()
    return area


def convex_hull(points):

    """

    Parameters
    ----------
    points : ndarray
             (n, 2) array of point coordinates

    Returns
    -------
    hull_poly : ogr
             an ogr polygon that is built out of
             the convex_hull

    """

    if isinstance(points, pd.DataFrame) :
        points = pd.DataFrame.as_matrix(points)

    hull = ConvexHull(points)
    return hull

def two_poly_overlap(poly1, poly2):
    """

    Parameters
    ----------
    poly1 : ogr polygon
            Any polygon that shares some kind of overlap
            with poly2

    poly2 : ogr polygon
            Any polygon that shares some kind of overlap
            with poly1

    Returns
    -------
     overlap_info : list
            The ratio convex hull volume / ideal_area

    """
    a_o = poly2.Intersection(poly1).GetArea()
    area1 = poly1.GetArea()
    area2 = poly2.GetArea()

    overlap_area = a_o
    overlap_percn = (a_o / (area1 + area2 - a_o)) * 100
    overlap_info = [overlap_percn, overlap_area]
    return overlap_info
