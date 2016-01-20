from collections import Hashable

import networkx as nx
import pandas as pd
import cv2
import numpy as np

from autocnet.control.control import C, POINT_TYPE, MEASURE_TYPE
from autocnet.fileio import io_json



class CandidateGraph(nx.Graph):
    """
    A NetworkX derived directed graph to store candidate overlap images.

    Parameters
    ----------

    Attributes
    ----------
    """

    def __init__(self,*args, **kwargs):
        super(CandidateGraph, self).__init__(*args, **kwargs)

    def add_image(self, identifier, *args, **kwargs):
        """
        Parameters
        ==========
        identifier : object
                     A Python hashable object to be used as the node key
        """
        if isinstance(identifier, Hashable):
            self.add_node(identifier, *args, **kwargs)
        else:
            raise TypeError('{} is not hashable and can not be a node id'.format(identifier))

    def adjacency_to_json(self, outputfile):
        """
        Write the edge structure to a JSON adjacency list

        Parameters
        ==========

        outputfile : str
                     PATH where the JSON will be written
        """
        adjacency_dict = {}
        for n in self.nodes():
            adjacency_dict[n] = self.neighbors(n)
        io_json.write_json(adjacency_dict, outputfile)

    def add_matches(self, source_node, matches):
        """

        Adds match data to a node and attributes the data to the
        appropriate edges, e.g. if A-B have a match, edge A-B is attributes
        with the pandas dataframe.

        Parameters
        ----------
        source_node : str
                      The identifier for the node

        matches : dataframe
                  The pandas dataframe containing the matches
        """

        #TODO: This really belongs in an outlier detection matcher class, not here.
        # Remove erroneous self neighbors
        matches = matches.loc[matches['matched_to'] != source_node]
        groups = matches.groupby('matched_to')
        for destination_node, group in groups:
            try:
                edge = self[source_node][destination_node]
            except:
                edge = self[destination_node][source_node]

            if 'matches' in edge.keys():
                df = edge['matches']
                edge['matches'] = pd.merge(df, matches, left_on='queryIdx', right_on='trainIdx')
            else:
                edge['matches'] = matches


    def compute_homography(self, source_key, destination_key, outlier_algorithm=cv2.RANSAC):
        """

        Parameters
        ----------
        source_key : str
                     The identifier for the source node
        destination_key : str
                          The identifier for the destination node
        Returns
        -------
         : tuple
           A tuple of the form (transformation matrix, bad entry mask)

        """
        #TODO: deal with no edge case / fail better
        #TODO: off-by-one error?
        if self.has_edge(source_key, destination_key):
            try:
                edge = self[source_key][destination_key]
            except:
                edge = self[destination_key][source_key]
            if 'matches' in edge.keys():
                source_keypoints = []
                destination_keypoints = []
                for i, row in edge['matches'].iterrows():
                    # Get the source and destination x,y coordinates for matches
                    source_idx = row['queryIdx_x']
                    src_keypoint = [self.node[source_key]['keypoints'][source_idx].pt[0],
                    self.node[source_key]['keypoints'][source_idx].pt[1]]

                    destination_idx = row['queryIdx_y']
                    dest_keypoint = [self.node[destination_key]['keypoints'][destination_idx-1].pt[0],
                                      self.node[destination_key]['keypoints'][destination_idx-1].pt[1]]

                    source_keypoints.append(src_keypoint)
                    destination_keypoints.append(dest_keypoint)

                return cv2.findHomography(np.array(source_keypoints), np.array(destination_keypoints),
                                          outlier_algorithm, 5.0)
            else:
                return ('', '')
        else:
            return ('','')

    def to_cnet(self):
        """
        Generate a control network (C) object from a graph

        Returns
        -------
        cnet : C
               A control network object
        """
        data = []
        point_ids = []
        serials = []
        for source, destination, attributes in self.edges_iter(data=True):
            for i, row in attributes['matches'].iterrows():

                # Get the source and destination x,y coordinates for matches
                source_idx = row['queryIdx_x']
                source_keypoints = (self.node[source]['keypoints'][source_idx].pt[0],
                                    self.node[source]['keypoints'][source_idx].pt[1])

                destination_idx = row['queryIdx_y']
                destination_keypoints = (self.node[destination]['keypoints'][destination_idx].pt[0],
                                         self.node[destination]['keypoints'][destination_idx].pt[1])

                data.append(source_keypoints)
                data.append(destination_keypoints)
                serials.append(source)
                serials.append(destination)
                point_ids.append(i)
                point_ids.append(i)

            point_types = [2] * len(point_ids)
            measure_types = [2] * len(point_ids)
            multi_index = pd.MultiIndex.from_tuples(list(zip(point_ids,point_types,
                                                             serials, measure_types)))

            columns = ['x', 'y']
            cnet = C(data, index=multi_index, columns=columns)

        # TODO: This method assumes a 2 image match and should be generalized to build an n-image C object

        return cnet

    @classmethod
    def from_adjacency(cls, inputfile):
        """
        Instantiate the class using an adjacency list

        Parameters
        ----------
        inputfile : str
                    The input file containing the graph representation

        Returns
        -------
         : object
           A Network graph object

        Examples
        --------
        >>> from autocnet.examples import get_path
        >>> inputfile = get_path('adjacency.json')
        >>> candidate_graph = network.CandidateGraph.from_adjacency(inputfile)
        """
        adjacency_dict = io_json.read_json(inputfile)
        return cls(adjacency_dict)


