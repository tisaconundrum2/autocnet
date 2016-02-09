import ogr
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
    intersection = ogr.CreateGeometryFromWkt(polys[0])
    for p in polys[1:]:
        geom = ogr.CreateGeometryFromWkt(p)
        intersection = intersection.Intersection(geom).ExportToWkt()

    area = intersection.GetArea()