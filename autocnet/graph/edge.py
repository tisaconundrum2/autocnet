from collections import MutableMapping
import warnings

import numpy as np
import pandas as pd
from pysal.cg.shapes import Polygon

from autocnet.matcher import subpixel as sp
from autocnet.matcher.homography import Homography
from autocnet.cg.cg import overlapping_polygon_area
from autocnet.vis.graph_view import plot_edge
from autocnet.matcher import outlier_detector as od
from autocnet.cg.cg import convex_hull_ratio


class Edge(dict, MutableMapping):
    """
    Attributes
    ----------
    source : hashable
             The source node

    destination : hashable
                  The destination node
    masks : set
            A list of the available masking arrays

    provenance : dict
                 With key equal to an autoincrementing integer and value
                 equal to a dict of parameters used to generate this
                 realization.
    """

    def __init__(self, source=None, destination=None):
        self.source = source
        self.destination = destination

        self._homography = None
        self._subpixel_offsets = None
        self.provenance = {}
        self._pid = 0

    def __repr__(self):
        return """
        Source Image Index: {}
        Destination Image Index: {}
        Available Masks: {}
        """.format(self.source, self.destination, self.masks)

    @property
    def masks(self):
        if not hasattr(self, '_masks'):
            self._masks = pd.Panel({self._pid: pd.DataFrame(index=self.matches.index)})
        return self._masks

    @masks.setter
    def masks(self, v):
        column_name = v[0]
        boolean_mask = v[1]
        current = self.masks[self._pid]
        current[column_name] = boolean_mask

    @property
    def error(self):
        if not hasattr(self, '_error'):
            self._error = pd.Panel({self._pid: pd.DataFrame(index=self.matches.index)})
        return self._error

    @error.setter
    def error(self, v):
        pass

    @property
    def homography(self):
        return self._homography

    @homography.setter
    def homography(self, v):
        self._homography = v

    def keypoints(self, clean_keys=[]):
        """
        Return a view of the keypoints dataframe after having applied some
        set of clean keys

        Parameters
        ----------
        clean_keys

        Returns
        -------

        """

        matches = self.matches

        # Build up a composite mask from all of the user specified masks
        if clean_keys:
            matches, _ = self._clean(clean_keys)

        # Now that we know the matches, build a pair of dataframes that are the truncated keypoints
        s_kps = self.source.keypoints.iloc[matches['source_idx']]
        d_kps = self.destination.keypoints.iloc[matches['destination_idx']]
        return s_kps, d_kps

    def symmetry_check(self):
        if hasattr(self, 'matches'):
            mask = od.mirroring_test(self.matches)
            self.masks = ('symmetry', mask)
        else:
            raise AttributeError('No matches have been computed for this edge.')

    def ratio_check(self, ratio=0.8):
        if hasattr(self, 'matches'):
            mask = od.distance_ratio(self.matches, ratio=ratio)
            self.masks = ('ratio', mask)
        else:
            raise AttributeError('No matches have been computed for this edge.')

    def compute_fundamental_matrix(self, clean_keys=[], **kwargs):

        if hasattr(self, 'matches'):
            matches = self.matches
        else:
            raise AttributeError('Matches have not been computed for this edge')

        if clean_keys:
            matches, mask = self._clean(clean_keys)

        s_keypoints = self.source.keypoints.iloc[matches['source_idx'].values]
        d_keypoints = self.destination.keypoints.iloc[matches['destination_idx'].values]

        transformation_matrix, fundam_mask = od.compute_fundamental_matrix(s_keypoints[['x', 'y']].values,
                                                                           d_keypoints[['x', 'y']].values,
                                                                           **kwargs)

        fundam_mask = fundam_mask.ravel()
        # Convert the truncated RANSAC mask back into a full length mask
        if clean_keys:
            mask[mask == True] = fundam_mask
        else:
            mask = fundam_mask
        self.masks = ('fundamental', mask)
        self.fundamental_matrix = transformation_matrix

    def compute_homography(self, method='ransac', clean_keys=[], pid=None, **kwargs):
        """
        For each edge in the (sub) graph, compute the homography
        Parameters
        ----------
        outlier_algorithm : object
                            An openCV outlier detections algorithm, e.g. cv2.RANSAC

        clean_keys : list
                     of string keys to masking arrays
                     (created by calling outlier detection)
        Returns
        -------
        transformation_matrix : ndarray
                                The 3x3 transformation matrix

        mask : ndarray
               Boolean array of the outliers
        """

        if hasattr(self, 'matches'):
            matches = self.matches
        else:
            raise AttributeError('Matches have not been computed for this edge')

        if clean_keys:
            matches, mask = self._clean(clean_keys)

        s_keypoints = self.source.keypoints.iloc[matches['source_idx'].values]
        d_keypoints = self.destination.keypoints.iloc[matches['destination_idx'].values]

        transformation_matrix, ransac_mask = od.compute_homography(s_keypoints[['x', 'y']].values,
                                                                   d_keypoints[['x', 'y']].values,
                                                                   **kwargs)

        ransac_mask = ransac_mask.ravel()
        # Convert the truncated RANSAC mask back into a full length mask
        if clean_keys:
            mask[mask == True] = ransac_mask
        else:
            mask = ransac_mask
        self.masks = ('ransac', mask)
        self.homography = Homography(transformation_matrix,
                                     s_keypoints[ransac_mask][['x', 'y']],
                                     d_keypoints[ransac_mask][['x', 'y']],
                                     index=mask[mask == True].index)

    def subpixel_register(self, clean_keys=[], threshold=0.8, upsampling=16,
                                 template_size=19, search_size=53):
        """
        For the entire graph, compute the subpixel offsets using pattern-matching and add the result
        as an attribute to each edge of the graph.

        Parameters
        ----------
        clean_keys : list
             of string keys to masking arrays
             (created by calling outlier detection)

        threshold : float
                    On the range [-1, 1].  Values less than or equal to
                    this threshold are masked and can be considered
                    outliers

        upsampling : int
                     The multiplier to the template and search shapes to upsample
                     for subpixel accuracy

        template_size : int
                        The size of the template in pixels, must be odd

        search_size : int
                      The size of the search
        """

        matches = self.matches
        self.subpixel_offsets = pd.DataFrame(0, index=matches.index, columns=['x_offset',
                                                                              'y_offset',
                                                                              'correlation',
                                                                              's_idx', 'd_idx'])

        # Build up a composite mask from all of the user specified masks
        if clean_keys:
            matches, mask = self._clean(clean_keys)

        # for each edge, calculate this for each keypoint pair
        for i, (idx, row) in enumerate(matches.iterrows()):
            s_idx = int(row['source_idx'])
            d_idx = int(row['destination_idx'])

            s_keypoint = self.source.keypoints.iloc[s_idx][['x', 'y']].values
            d_keypoint = self.destination.keypoints.iloc[d_idx][['x', 'y']].values

            # Get the template and search window
            s_template = sp.clip_roi(self.source.handle, s_keypoint, template_size)
            d_search = sp.clip_roi(self.destination.handle, d_keypoint, search_size)

            try:
                x_off, y_off, strength = sp.subpixel_offset(s_template, d_search, upsampling=upsampling)
                self.subpixel_offsets.loc[idx] = [x_off, y_off, strength,s_idx, d_idx ]
            except:
                warnings.warn('Template-Search size mismatch, failing for this correspondence point.')
                continue

        self.subpixel_offsets.to_sparse(fill_value=0.0)

        # Compute the mask for correlations less than the threshold
        mask = self.subpixel_offsets['correlation'] >= threshold

        self.masks = ('subpixel', mask)

    def coverage_ratio(self, clean_keys=[]):
        """
        Compute the ratio $area_{convexhull} / area_{imageoverlap}$.

        Returns
        -------
        ratio : float
                The ratio $area_{convexhull} / area_{imageoverlap}$
        """
        if self.homography is None:
            raise AttributeError('A homography has not been computed. Unable to determine image overlap.')

        matches = self.matches
        # Build up a composite mask from all of the user specified masks
        if clean_keys:
            matches, _ = self._clean(clean_keys)

        d_idx = matches['destination_idx'].values
        keypoints = self.destination.keypoints.iloc[d_idx][['x', 'y']].values
        if len(keypoints) < 3:
            raise ValueError('Convex hull computation requires at least 3 measures.')

        source_geom, proj_geom, ideal_area = self.compute_homography_overlap()

        ratio = convex_hull_ratio(keypoints, ideal_area)
        return ratio

    def compute_homography_overlap(self):
        """
        Using the homography, estimate the overlapping area
        between images on the edge

        Returns
        -------
        source_geom : object
                      PySAL Polygon object of the source pixel bounding box

        projected_geom : object
                         PySAL Polygon object of the destination geom projected
                         into the source reference system using the current
                         homography

        area : float
               The estimated area
        """

        source_geom = self.source.handle.pixel_polygon
        destination_geom = self.destination.handle.pixel_polygon

        # Project using the homography
        vertices_to_project = destination_geom.vertices
        for i, v in enumerate(vertices_to_project):
            vertices_to_project[i] = tuple(np.array([v[0], v[1], 1]).dot(self.homography)[:2])
        projected_geom = Polygon(vertices_to_project)

        # Estimate the overlapping area
        area = overlapping_polygon_area([source_geom, projected_geom])

        return source_geom, projected_geom, area

    def plot(self, ax=None, clean_keys=[], **kwargs):
        return plot_edge(self, ax=ax, clean_keys=clean_keys, **kwargs)

    def _clean(self, clean_keys, pid=None):
        """
        Given a list of clean keys and a provenance id compute the
        mask of valid matches

        Parameters
        ----------
        clean_keys : list
                     of columns names (clean keys)
        pid : int
              The provenance id of the parameter set to be cleaned.
              Defaults to the last run.

        Returns
        -------
        matches : dataframe
                  A masked view of the matches dataframe

        mask : series
                    A boolean series to inflate back to the full match set
        """
        if not pid:
            pid = self._pid
        panel = self.masks[pid]
        mask = panel[clean_keys].all(axis=1)
        matches = self.matches[mask]
        return matches, mask