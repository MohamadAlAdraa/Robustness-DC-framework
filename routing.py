import networkx as nx
from collections import defaultdict
from itertools import islice


class Routing():
    def __init__(self, G):
        self.G = G

    def ksp(self, k):
        paths = defaultdict(defaultdict)
        # i=0
        for src in self.G.nodes():
            # i += 1
            for dst in self.G.nodes():
                if src != dst:
                    paths[src][dst] = self.k_shortest_paths(src, dst, k)
            # if src != dst:
            #     print(i, len(paths[src][dst]))
        return paths

    def k_shortest_paths(self, source, target, k=4, weight=None):
        return list(islice(nx.shortest_simple_paths(self.G, source, target, weight=weight), k))
        # return list(nx.shortest_simple_paths(self.G, source, target, weight=weight))
