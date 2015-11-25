from collections import Hashable
from networkx import MultiDiGraph

class CandidateGraph(MultiDiGraph):
    def __init__(self,*args, **kwargs):
        super(CandidateGraph,self).__init__(*args, **kwargs)

    def add_image(self, identifier):
        if isinstance(identifier, Hashable):
            self.add_node(identifier)
        else:
            raise TypeError('{} is not hashable and can not be a node id'.format(identifier))
