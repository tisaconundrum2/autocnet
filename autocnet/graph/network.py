from collections import Hashable
from networkx import DiGraph

class CandidateGraph(DiGraph):
    def __init__(self,*args, **kwargs):
        super(CandidateGraph,self).__init__(*args, **kwargs)

    def add_image(self, identifier):
        """
        Parameters
        ==========
        identifier : object
                     A Python hashable object to be used as the node key
        """
        if isinstance(identifier, Hashable):
            self.add_node(identifier)
        else:
            raise TypeError('{} is not hashable and can not be a node id'.format(identifier))
