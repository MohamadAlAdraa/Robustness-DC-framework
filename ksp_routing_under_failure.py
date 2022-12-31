from collections import defaultdict
from ksp_routing import read_paths_from_file, find_ksp_throughput, tt
from utilities import read_graph_from_file
import math
import networkx as nx
import random
import json


strat_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d24/'
xpander_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_xpander/d24/'
jellyfish_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_jellyfish/d24/'
G = read_graph_from_file(strat_path + 'StratAL2048_8.txt')
G1 = read_graph_from_file(xpander_path + 'XpanderAL2048_8.txt')
G2 = read_graph_from_file(jellyfish_path + 'JellyfishAL2048_8.txt')


strat_paths20 = read_paths_from_file('strat20.json')
xpander_paths20 = read_paths_from_file('xpander20.json')
jellyfish_paths20 = read_paths_from_file('jellyfish20.json')


flows_per_switch = 96
fail_rate = [0.01,0.03,0.06,0.09,0.12,0.15,0.18,0.21,0.24,0.27,0.3]
alpha_f = 0.05
topos_name = ['xpander', 'jellyfish', 'strat']
topos_graph = {'strat': G, 'xpander':G1, 'jellyfish':G2}
topos_path = {'strat': strat_paths20, 'xpander':xpander_paths20, 'jellyfish':jellyfish_paths20}
ks = [6]
data_to_be_written_to_the_file = defaultdict(defaultdict)
for k in ks:
    for topo_name in topos_name:
        print(topo_name)
        data_to_be_written_to_the_file = defaultdict(defaultdict)
        util = dict()
        for iter in range(5):
            print(iter)
            alpha, num_of_b_c, num_of_ac_c, accepted_connections, blocked_connections, traffic_matrix, paths_of_accepted_connections, utilization, connetions_on_edges = find_ksp_throughput(topos_graph[topo_name], k, tt(flows_per_switch, topos_graph[topo_name]), alpha_f, util, fp=topos_path[topo_name])
            paths_of_accepted_connections_ = dict()
            for key, value in paths_of_accepted_connections.items():
                paths_of_accepted_connections_[str(key)] = paths_of_accepted_connections[key]
            data_to_be_written_to_the_file[str(iter)]['initial'] = {'paths_of_accepted_connections':paths_of_accepted_connections_, 'num_of_ac_c': num_of_ac_c, 'num_of_b_c': num_of_b_c, 'max_util': max(list(utilization.values())), 'min_util': min(list(utilization.values())), 'sum_util': sum(list(utilization.values()))}
            for rate in fail_rate:
                print(rate, end=',')
                temp_utilization = utilization.copy()
                number_of_edges_to_be_removed = math.floor(rate * len(list(topos_graph[topo_name].edges())))
                print(number_of_edges_to_be_removed)
                b = True
                G_after_failure = nx.Graph()
                edges_to_be_removed_after_failure = None
                while (b):
                    G_temp = topos_graph[topo_name].copy()
                    edges_to_be_removed = random.sample(topos_graph[topo_name].edges(), k=number_of_edges_to_be_removed)
                    edges_to_be_removed_after_failure = edges_to_be_removed.copy()
                    for rm in edges_to_be_removed:
                        G_temp.remove_edge(rm[0], rm[1])
                    if nx.is_connected(G_temp):
                        b = False
                        G_after_failure = G_temp.copy()
                temp_edges = []

                for e in edges_to_be_removed_after_failure:
                    xx,xxx = e
                    temp_edges.append((str(min(int(xx), int(xxx))), str(max(int(xx), int(xxx)))))
                edges_to_be_removed_after_failure = temp_edges.copy()

                affected_connections = []
                for e_f in edges_to_be_removed_after_failure:
                    if e_f in connetions_on_edges:
                        for connection in connetions_on_edges[e_f]:
                            if connection not in affected_connections:
                                affected_connections.append(connection)

                for connection in affected_connections:
                    for y in range(len(paths_of_accepted_connections[connection]) - 1):
                        x1 = paths_of_accepted_connections[connection][y]
                        x2 = paths_of_accepted_connections[connection][y+1]
                        edge_name = (str(min(int(x1), int(x2))), str(max(int(x1), int(x2))))
                        temp_util_value = float("{:.2f}".format(float("{:.2f}".format(temp_utilization[edge_name])) - float("{:.2f}".format(alpha))))
                        temp_utilization[edge_name] = temp_util_value
                alpha_temp, num_of_b_c_temp, num_of_ac_c_temp, accepted_connections_temp, blocked_connections_temp, traffic_matrix_temp, paths_of_accepted_connections_temp, utilization_temp, connetions_on_edges_temp = find_ksp_throughput(G_after_failure, k, affected_connections, alpha, temp_utilization, k=20)
                paths_of_accepted_connections__ = dict()
                for key, value in paths_of_accepted_connections_temp.items():
                    paths_of_accepted_connections__[str(key)] = paths_of_accepted_connections_temp[key]
                data_to_be_written_to_the_file[str(iter)][str(rate)] = {'paths_of_accepted_connections': paths_of_accepted_connections__,'blocked_connections_temp': blocked_connections_temp, 'num_of_ac_c': num_of_ac_c_temp, 'num_of_b_c': num_of_b_c_temp, 'max_util': max(list(utilization_temp.values())), 'min_util': min(list(utilization_temp.values())), 'sum_util': sum(list(utilization_temp.values()))}
            print('\n')
        with open("ksp_data_to_be_used_"+ topo_name + str(flows_per_switch) + "_v4" + ".json", 'w') as f:
            json.dump(data_to_be_written_to_the_file, f)


