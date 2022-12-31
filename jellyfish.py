import random
import networkx as nx


class Jellyfish:
    G = nx.Graph()

    def __init__(self, number_of_sw, d):
        self.number_of_sw = number_of_sw
        self.d = d
        self.G = nx.Graph()
        self.G = nx.random_regular_graph(d, number_of_sw)
        # self.G = nx.random_regular_graph(d, d + 1)
        # dict_d = dict.fromkeys(range(0, d + 1), d)
        # for i in range(d + 1):
        #     dict_d[i] -= d
        # for g in range(d + 1, number_of_sw):
        #     self.G.add_node(g)
        #     dict_d[g] = d
        #     nodes_to_be_connected = [i for i, j in dict_d.items() if j > 0 and i != g]
        #     if len(nodes_to_be_connected) != 0:
        #         self.G.add_edge(g, nodes_to_be_connected[0])
        #         dict_d[g] -= 1
        #         dict_d[nodes_to_be_connected[0]] -= 1
        #     rr = dict_d[g] // 2
        #     for i in range(rr):
        #         neighbors = list(self.G.neighbors(g))
        #         if len(neighbors) == 0:
        #             e = random.sample(self.G.edges(), k=1)
        #             self.G.remove_edge(e[0][0], e[0][1])
        #             self.G.add_edge(e[0][0], g)
        #             self.G.add_edge(e[0][1], g)
        #             dict_d[g] -= 2
        #         else:
        #             filtered_edges = list(self.G.edges()).copy()
        #             for e in self.G.edges():
        #                 for neighbor in neighbors:
        #                     if neighbor in e:
        #                         filtered_edges.remove(e)
        #                         break
        #             ed = random.sample(filtered_edges, k=1)
        #             self.G.remove_edge(ed[0][0], ed[0][1])
        #             self.G.add_edge(ed[0][0], g)
        #             self.G.add_edge(ed[0][1], g)
        #             dict_d[g] -= 2


class Jellyfish_Topology:
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
