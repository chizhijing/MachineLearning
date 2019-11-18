import networkx as nx
from networkx import Graph

class PaperGraph(Graph):
    def __init__(self,node_info_dict):
        self.node_info_dict=node_info_dict
        self.add_nodes_from(node_info_dict.keys())


if __name__=='__main__':

    p_graph=PaperGraph()