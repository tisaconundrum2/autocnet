import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull

from autocnet.utils import utils

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


def convex_hull(points):

    """

    Parameters
    ----------
    points : ndarray
             (n, 2) array of point coordinates

    Returns
    -------
    hull : 2-D convex hull
            Provides a convex hull that is used
            to determine coverage

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
    overlap_percn : float
                    The percentage of image overlap

    overlap_area : float
                   The total area of overalap

    """
    overlap_area_polygon = poly2.Intersection(poly1)
    overlap_area = overlap_area_polygon.GetArea()
    area1 = poly1.GetArea()
    area2 = poly2.GetArea()

    overlap_percn = (overlap_area / (area1 + area2 - overlap_area)) * 100
    return overlap_percn, overlap_area, overlap_area_polygon


def get_area(poly1, poly2):
    """

    Parameters
    ----------
    poly1 : ogr polygon
            General ogr polygon

    poly2 : ogr polygon
            General ogr polygon

    Returns
    -------
    intersection_area : float
                        returns the intersection area
                        of two polygons

    """
    intersection_area = poly1.Intersection(poly2).GetArea()
    return intersection_area


def compute_vor_weight(vor, voronoi_df, intersection_poly, verbose):
    """

    Parameters
    ----------
    vor : Voronoi
          Scipy Voronoi object

    voronoi_df : dataframe
                 3 column pandas dataframe of x, y, and weights

    intersection_poly : polygon
                        Intersection polygon to use for
                        clipping the voronoi diagram

    verbose : boolean
              Set to True to display the calculated voronoi diagram
              to the user
    """
    i = 0
    poly_array = []
    for region in vor.regions:
        region_point = vor.points[np.argwhere(vor.point_region==i)]
        if -1 not in region:
            polygon_points = [vor.vertices[i] for i in region]
            if len(polygon_points) != 0:
                polygon = utils.array_to_poly(polygon_points)
                intersection = polygon.Intersection(intersection_poly)
                poly_array = np.append(poly_array, intersection)
                polygon_area = intersection.GetArea()
                voronoi_df.loc[(voronoi_df["x"] == region_point[0][0][0]) &
                               (voronoi_df["y"] == region_point[0][0][1]),
                               'weights'] = polygon_area
        i += 1
