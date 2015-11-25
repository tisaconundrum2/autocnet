from networkx import MultiDiGraph

class CandidateGraph(MultiDiGraph):
    def __init__(self,*args, **kwargs):
        super(CandidateGraph,self).__init__(*args, **kwargs)

    def add_image(identifier):
        self.add_node(identifier)

    def add_edge(origin, destination):
        """
        origin : int
                 The origin node id
        destination : int
                      The destination node id
        """
        pass
