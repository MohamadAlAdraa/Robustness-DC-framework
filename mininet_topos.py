from mininet.net import Mininet
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.node import RemoteController
from utilities import draw_graph, get_the_best_xpander, find_average_shortest_path, get_the_best_jellyfish
import fattree
import xpander
import strat
import jellyfish
import networkx as nx


class Topology:
	switches = {}
	hosts = {}
	linkOpt = dict(bw=1000, delay='0.0ms', loss=0, max_queue_size=1000, use_htb=False, use_tbf=True)
	hostsIps = {}
	net = Mininet(link=TCLink)
	fatTree_graph = nx.Graph()
	xpander_graph = nx.Graph()
	strat_graph = nx.Graph()
	jellyfish_graph = nx.Graph()
	CONTROLLER_IP = "172.17.0.1"

	def __init__(self, topo_name):
		self.topo_name = topo_name

	def create_topo(self):
		if self.topo_name == "fattree":
			###########################
			# To be determined
			# k
			############################
			k = 4
			obj = fattree.Fattree(k, self.net, self.linkOpt)
			self.switches = obj.switches
			self.hosts = obj.hosts
			self.hostsIps = obj.hostsIps
			self.fatTree_graph = obj.G
		elif self.topo_name == "xpander":
			###########################
			# To be determined
				# number_of_hosts, number_of_hosts_per_switch, d_regular, number_of_iterations, lift_k
			############################
			num_switches = 16
			number_of_hosts = 64
			number_of_hosts_per_switch = 4
			d_regular = 2
			self.xpander_graph = get_the_best_xpander(num_switches, number_of_hosts, number_of_hosts_per_switch, d_regular)
			# obj = xpander.Xpander_Topology(self.xpander_graph, number_of_hosts_per_switch, self.net, self.linkOpt)
			# self.switches = obj.switches
			# self.hosts = obj.hosts
			# self.hostsIps = obj.hostsIps
		elif self.topo_name == "strat":
			###########################
			# To be determined
				# filename for the adjacency list, number_of_hosts_per_switch
			############################
			filename = './ad_list_strat/StratAL32.txt'
			number_of_hosts_per_switch = 1
			obj = strat.Strat(filename)
			self.strat_graph = obj.G
			obj1 = strat.Strat_Topology(self.strat_graph, number_of_hosts_per_switch, self.net, self.linkOpt)
			self.switches = obj1.switches
			self.hosts = obj1.hosts
			self.hostsIps = obj1.hostsIps
		elif self.topo_name == "jellyfish":
			###########################
			# To be determined
			# number_of_hosts, number_of_hosts_per_switch, d_regular
			############################
			number_of_hosts_per_switch = 4
			number_of_switches = 16
			d_regular = 3
			self.jellyfish_graph = get_the_best_jellyfish(number_of_switches, d_regular)
			obj1 = jellyfish.Jellyfish_Topology(self.jellyfish_graph, number_of_hosts_per_switch, self.net, self.linkOpt)
			self.switches = obj1.switches
			self.hosts = obj1.hosts
			self.hostsIps = obj1.hostsIps
		return self.net

	def connect_to_controller(self):
		self.net.addController('controller', controller=RemoteController, ip=self.CONTROLLER_IP)


if __name__ == "__main__":
	print('Nothing to be executed')
	# t = Topology('xpander')
	# net = t.create_topo()
	# draw_graph(t.jellyfish_graph, 'jel')
	# print(find_average_shortest_path(t.jellyfish_graph))
	# G = nx.read_adjlist("test.adjlist")
	# draw_graph(G, 'jel1')
	# print(find_average_shortest_path(G))
	# t.connect_to_controller()
	# net.start()
	# CLI(net)
	# net.stop()


