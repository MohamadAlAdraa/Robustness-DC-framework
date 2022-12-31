import networkx as nx
#import logging


# logging.basicConfig(filename='./fattree.log', level=logging.INFO)
# logger = logging.getLogger(__name__)


class Fattree:
    # logger.debug("Class Fattree")
    hosts = {}
    switches = {}
    hostsIps = {}
    agg_sw_names = []
    G = nx.Graph()

    def __init__(self, k, net, linkOpt):
        # logger.debug("Class Fattree init")
        self.k = k
        self.net = net
        self.linkOpt = linkOpt
        
        pods = [self.make_pod(i) for i in range(self.k)]
        
        for core_num in range((self.k//2)**2):
            s = self.net.addSwitch('c_s%d' % core_num, dpid='0000000010%02x0000' % core_num, protocols="OpenFlow13")
            self.switches['c_s%d' % core_num] = s
            self.G.add_node('c_s%d' % core_num)
            stride_num = core_num // (k//2)
            for i in range(self.k):
                self.net.addLink(s, pods[i][stride_num], **self.linkOpt)
                self.G.add_edge('c_s%d' % core_num, self.agg_sw_names[i][stride_num])

    def make_pod(self, pod_num):
        
        lower_layer_switches = []
        edge_switches = []
        for i in range(self.k // 2):
            s = self.net.addSwitch('p%d_s%d' % (pod_num, i), dpid='000000002000%02x%02x' % (pod_num, i), protocols="OpenFlow13")
            self.switches['p%d_s%d' % (pod_num, i)] = s
            self.G.add_node('p%d_s%d' % (pod_num, i))
            lower_layer_switches.append(s)
            edge_switches.append('p%d_s%d' % (pod_num, i))

        for i, switch in enumerate(lower_layer_switches):
            for j in range(2, self.k // 2 + 2):
                h = self.net.addHost('p%d_s%d_h%d' % (pod_num, i, j-1), ip='10.%d.%d.%d' % (pod_num, i, j-1))
                self.hosts['p%d_s%d_h%d' % (pod_num, i, j-1)] = h
                self.hostsIps['p%d_s%d_h%d' % (pod_num, i, j-1)] = '10.%d.%d.%d' % (pod_num, i, j-1)
                self.net.addLink(switch, h, **self.linkOpt)

        upper_layer_switches = [] 
        agg_switches = []
        for i in range(self.k//2, self.k):
            s = self.net.addSwitch('p%d_s%d' % (pod_num, i), dpid='000000002000%02x%02x' % (pod_num, i), protocols="OpenFlow13")
            self.switches['p%d_s%d' % (pod_num, i)] = s
            self.G.add_node('p%d_s%d' % (pod_num, i))
            upper_layer_switches.append(s)
            agg_switches.append('p%d_s%d' % (pod_num, i))

        self.agg_sw_names.append(agg_switches)

        for lower in lower_layer_switches:
            for upper in upper_layer_switches:
                self.net.addLink(lower, upper, **self.linkOpt)

        for l in edge_switches:
            for u in agg_switches:
                self.G.add_edge(l, u)

        return upper_layer_switches


