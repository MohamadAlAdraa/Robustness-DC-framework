import networkx as nx
import math
import random
#import logging


# logging.basicConfig(filename='./xpander.log', level=logging.INFO)
# logger = logging.getLogger(__name__)

class Xpander:
    # logger.debug("Class Xpander")
    Initial_G = nx.Graph()
    lifted_graph = nx.Graph()
    not_lifted_graph = nx.Graph()
    deterministic_lifted_graph = nx.Graph()
    incremental_graph = nx.Graph()
    highest_spectral_gap = 0

    def __init__(self, number_of_switches, number_of_hosts, number_of_hosts_per_switch, d_regular, c_G=None, spectral_gap='a', k_lift=2):
        # logger.debug("Class Xpander init")
        self.number_of_hosts = number_of_hosts
        self.c_G = c_G
        self.number_of_hosts_per_switch = number_of_hosts_per_switch
        self.d_regular = d_regular
        self.k_lift = k_lift
        self.number_of_switches = number_of_switches
        # it can take two values only "a" or "l"
        self.spectral_gap = spectral_gap
        ################################################################################################################
        # Apply the incremental algorithm
        self.Initial_G = nx.random_regular_graph(self.d_regular, self.d_regular+1)
        # self.Initial_G = self.c_G.copy()
        self.incremental_graph = self.incremental_expansion_algo(self.Initial_G, number_of_switches)
        ################################################################################################################

        ################################################################################################################
        # number_of_switches, number_of_lifts, Initial_G = self.graph_initialization()
        # self.Initial_G = Initial_G
        # Apply the deterministic algorithm after each lifted graph
        # for i in range(number_of_lifts):
        #     self.not_lifted_graph, self.lifted_graph = self.two_lifting_algorithm(self.Initial_G)
        #    # self.Initial_G = self.lifted_graph.copy()
        #   # self.deterministic_lifted_graph = self.deterministic_two_lift(self.not_lifted_graph, self.lifted_graph)
        #   # self.Initial_G = self.deterministic_lifted_graph.copy()
        # self.deterministic_lifted_graph = self.deterministic_two_lift(self.not_lifted_graph, self.lifted_graph)
        ################################################################################################################

    def graph_initialization(self):
        number_of_switches = int(math.ceil(self.number_of_hosts/self.number_of_hosts_per_switch))
        number_of_lifts = int(math.ceil(math.log(number_of_switches/(self.d_regular+1), self.k_lift)))
        Initial_G = nx.random_regular_graph(self.d_regular, self.d_regular+1)
        return number_of_switches, number_of_lifts, Initial_G

    def two_lifting_algorithm(self, G):
        new_nodes_to_be_added = [2*i for i in G.nodes()]
        combine_new_nodes_with_old_nodes = dict(zip(G.nodes(), new_nodes_to_be_added))
        lifted_graph_ = nx.relabel_nodes(G, combine_new_nodes_with_old_nodes)
        not_lifted_graph = nx.relabel_nodes(G, combine_new_nodes_with_old_nodes)
        previous_nodes = lifted_graph_.nodes()
        new_nodes = [n + 1 for n in previous_nodes]
        for n in new_nodes:
            lifted_graph_.add_node(n)
            not_lifted_graph.add_node(n)
        edgelist = list(lifted_graph_.edges())
        enumeration_list = [i for i in range(1, len(edgelist) + 1)]
        random.shuffle(enumeration_list)
        c = 0
        for u, v in edgelist:
            lifted_graph_.remove_edge(u, v)
            # || MATCHING
            if enumeration_list[c] % 2 != 0:
                lifted_graph_.add_edge(u, v)
                lifted_graph_.add_edge(u + 1, v + 1)
            # X MATCHING
            else:
                lifted_graph_.add_edge(u, v + 1)
                lifted_graph_.add_edge(u + 1, v)
            c += 1
        return not_lifted_graph, lifted_graph_

    def find_spectral_gap(self, G):
        if self.spectral_gap == "a":
            e = nx.adjacency_spectrum(G)
            flat = abs(e.flatten())
            flat.sort()
            largest = flat[-1]
            secondLargest = flat[-2]
            return largest - secondLargest
        elif self.spectral_gap == "l":
            e = nx.laplacian_spectrum(G)
            flat = e.flatten()
            flat.sort()
            return flat[1]

    def find_smallest_eigen_value_of_lap(self, G):
        e = nx.laplacian_spectrum(G)
        flat = e.flatten()
        flat.sort()
        return flat[1]

    def sort_dict_by_value_highest_to_lowest(self, dict):
        return {k: v for k, v in sorted(dict.items(),reverse=True,  key=lambda item: item[1])} #

    def deterministic_two_lift(self, G, lifted_G):
        two_criss_cross_graph = lifted_G.copy()
        improved = True
        i=0
        while improved:
            improved = False
            two_criss_cross_graph_to_be_improved = two_criss_cross_graph.copy()
            Initial_G_edges = list(G.edges())
            two_criss_cross_graph_to_be_improved_edges = list(two_criss_cross_graph_to_be_improved.edges())
            for u, v in Initial_G_edges:
                if (u, v) in two_criss_cross_graph_to_be_improved_edges:
                    two_criss_cross_graph_to_be_improved.remove_edge(u, v)
                    two_criss_cross_graph_to_be_improved.remove_edge(u+1, v+1)
                    two_criss_cross_graph_to_be_improved.add_edge(u, v+1)
                    two_criss_cross_graph_to_be_improved.add_edge(u+1, v)
                else:
                    two_criss_cross_graph_to_be_improved.remove_edge(u, v+1)
                    two_criss_cross_graph_to_be_improved.remove_edge(u+1, v)
                    two_criss_cross_graph_to_be_improved.add_edge(u, v)
                    two_criss_cross_graph_to_be_improved.add_edge(u+1, v+1)
                old_spectral_gap = self.find_spectral_gap(two_criss_cross_graph)
                new_spectral_gap = self.find_spectral_gap(two_criss_cross_graph_to_be_improved)
                if new_spectral_gap > old_spectral_gap:
                    self.highest_spectral_gap = new_spectral_gap
                    two_criss_cross_graph = two_criss_cross_graph_to_be_improved.copy()
                    improved = True
            i += 1
        return two_criss_cross_graph

    def incremental_expansion_algo(self, G, number_of_switches):
        G1 = G.copy()
        N = number_of_switches - (len(list(G1.nodes())))
        d = self.d_regular
        T = len(list(G1.nodes()))
        while len(list(G.nodes())) < N + T:
            GapMap = {}
            Q = []
            for (u, v) in G.edges():
                G.remove_edge(u, v)
                uvGap = self.find_smallest_eigen_value_of_lap(G)
                # uvGap= self.find_spectral_gap(G)
                G.add_edge(u, v)
                GapMap[(u, v)] = uvGap
            GapMap = self.sort_dict_by_value_highest_to_lowest(GapMap)
            for u, v in GapMap.keys():
                if len(Q) == d:
                    break
                elif (u not in Q) and (v not in Q):
                    Q.append(u)
                    Q.append(v)
                    G.remove_edge(u, v)
            new_n = len(G.nodes())
            G.add_node(new_n)
            for i in Q:
                G.add_edge(new_n, i)
        return G


class Xpander_Topology:
    hosts = {}
    switches = {}
    hostsIps = {}

    def __init__(self, G, number_of_hosts_per_switch, net, linkOpt):
        self.G = G
        self.net = net
        self.linkOpt = linkOpt
        self.number_of_hosts_per_switch = number_of_hosts_per_switch

        for n in G.nodes():
            s = net.addSwitch('s%d' % n, protocols="OpenFlow13")
            self.switches['s%d' % n] = s
            for i in range(number_of_hosts_per_switch):
                h = net.addHost('s%d_h%d' % (n, i+1), ip='10.0.%d.%d' % (n, i+1))
                self.hosts['s%d_h%d' % (n, i+1)] = h
                self.hostsIps['s%d_h%d' % (n, i+1)] = '10.0.%d.%d' % (n, i+1)
                net.addLink(s, h, **linkOpt)

        for e in G.edges():
            net.addLink(self.switches.get('s%d' % e[0]), self.switches.get('s%d' % e[1]), **linkOpt)










