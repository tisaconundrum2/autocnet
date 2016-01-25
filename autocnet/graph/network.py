import os

import networkx as nx
import pandas as pd
import cv2
import numpy as np

from autocnet.control.control import C
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
        self.node_counter = 0
        node_labels = {}

        for node_name, node_attributes in self.nodes_iter(data=True):
            if os.path.isabs(node_name):
                node_attributes['image_name'] = os.path.basename(node_name)
                node_attributes['image_path'] = node_name
            else:
                node_attributes['image_name'] = node_name
                node_attributes['image_path'] = None

            node_labels[node_attributes['image_name']] = self.node_counter
            self.node_counter += 1

        nx.relabel_nodes(self, node_labels, copy=False)

    def add_image(self, *args, **kwargs):
        """
        Parameters
        ==========
        """

        self.add_node(self.node_counter, *args, **kwargs)
        self.node_counter += 1

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

    def add_matches(self, matches):
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

        source_groups = matches.groupby('source_image')
        for i, source_group in source_groups:
            for j, dest_group in source_group.groupby('destination_image'):
                source_key = dest_group['source_image'].values[0]
                destination_key = dest_group['destination_image'].values[0]
                try:
                    edge = self[source_key][destination_key]
                except:
                    edge = self[destination_key][source_key]

                if 'matches' in edge.keys():
                    df = edge['matches']
                    edge['matches'] = pd.concat([df, dest_group])
                else:
                    edge['matches'] = dest_group

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
           The returned tuple is empty if there is no edge between the source and destination nodes or
           if it exists, but has not been populated with a matches dataframe.

        """
        if self.has_edge(source_key, destination_key):
            try:
                edge = self[source_key][destination_key]
            except:
                edge = self[destination_key][source_key]
            if 'matches' in edge.keys():
                source_keypoints = []
                destination_keypoints = []

                for i, row in edge['matches'].iterrows():
                    source_idx = row['source_idx']
                    src_keypoint = [self.node[source_key]['keypoints'][int(source_idx)].pt[0],
                                    self.node[source_key]['keypoints'][int(source_idx)].pt[1]]
                    destination_idx = row['destination_idx']
                    dest_keypoint = [self.node[destination_key]['keypoints'][int(destination_idx)].pt[0],
                                     self.node[destination_key]['keypoints'][int(destination_idx)].pt[1]]

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
        merged_cnet : C
                      A control network object
        """
        merged_cnet = None

        for source, destination, attributes in self.edges_iter(data=True):
            matches = attributes['matches']
            kp1 = self.node[source]['keypoints']
            kp2 = self.node[destination]['keypoints']

            pt_idx = 0
            values = []
            for idx, row in matches.iterrows():
                # Composite matching key (node_id, point_id)
                m1 = (source, int(row['source_idx']))
                m2 = (destination, int(row['destination_idx']))

                values.append([kp1[m1[1]].pt[0],
                               kp1[m1[1]].pt[1],
                               m1,
                               pt_idx,
                               source])

                values.append([kp2[m2[1]].pt[0],
                               kp2[m2[1]].pt[1],
                               m2,
                               pt_idx,
                               destination])
                pt_idx += 1

            columns = ['x', 'y', 'idx', 'pid', 'nid']

            cnet = C(values, columns=columns)

            if merged_cnet is None:
                merged_cnet = cnet.copy(deep=True)
            else:
                pid_offset = merged_cnet['pid'].max() + 1  # Get the current max point index
                cnet[['pid']] += pid_offset

                # Inner merge on the dataframe identifies common points
                common = pd.merge(merged_cnet, cnet, how='inner', on='idx', left_index=True, suffixes=['_r',
                                                                                                      '_l'])

                # Iterate over the points to be merged and merge them in.
                for i, r in common.iterrows():
                    new_pid = r['pid_r']
                    update_pid = r['pid_l']
                    cnet.loc[cnet['pid'] == update_pid, ['pid']] = new_pid  # Update the point ids

                # Perform the concat
                merged_cnet = pd.concat([merged_cnet, cnet])
                merged_cnet.drop_duplicates(['idx', 'pid'], keep='first', inplace=True)

        return merged_cnet

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


