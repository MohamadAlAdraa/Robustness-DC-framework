import networkx as nx
from utilities import write_graph_to_file, read_graph_from_file

class Strat:
    G = nx.Graph()

    def __init__(self, filename):
        self.filename = filename
        self.G = nx.Graph()

        f = open(filename, "r")
        l = f.readlines()
        f.close()

        for i in range(len(l)):
            self.G.add_node(i + 1)

        for i in range(len(l)):
            x = l[i].split(" ")
            for j in x:
                j = j.strip('\n')
                if int(j) < (i + 1):
                    self.G.add_edge(i + 1, int(j))


class Strat_Topology:
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

