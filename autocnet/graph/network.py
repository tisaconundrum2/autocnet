from collections import Hashable
from networkx import DiGraph

from autocnet.fileio import io_json
class CandidateGraph(DiGraph):
    #TODO: This would be better with composition, and then dispatch the 
    # network X calls to the graph object.

    def __init__(self,*args, **kwargs):
        super(CandidateGraph,self).__init__(*args, **kwargs)

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

    @classmethod
    def from_adjacency(cls, inputfile):
        """
        Instantiate the class using an adjacency list

        Parameters
        ==========
        inputfile : str
                    The input file containing the graph representation
        """
        #TODO: This is better as a generic reader that tries drivers until 
        # a valid dict is returned.
        adjacency_dict = io_json.read_json(inputfile)
        return cls(adjacency_dict)

