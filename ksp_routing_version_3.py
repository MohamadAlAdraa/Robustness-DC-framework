import random

import networkx as nx
import numpy as np

from routing import Routing
from utilities import read_graph_from_file, lb_av_shortest_path, write_graph_to_file
import json


def write_paths_to_file(G, k, filename):
    r = Routing(G)
    paths = r.ksp(k)
    with open(filename + '.json', 'w') as f:
        json.dump(paths, f)


def read_paths_from_file(filename):
    f = open(filename)
    data = json.load(f)
    return data


def random_permutation_traffic_helper(G):
    traffic_matrix_random_one_to_one = []
    nodes = list(G.nodes())
    b = True
    while b:
        temp_nodes = list(G.nodes())
        random.shuffle(temp_nodes)
        b = False
        for i in range(len(nodes)):
            if nodes[i] == temp_nodes[i]:
                b = True
                break
    for i in range(len(nodes)):
        traffic_matrix_random_one_to_one.append((nodes[i], temp_nodes[i]))
    return traffic_matrix_random_one_to_one


def random_permutation_traffic(G, num_of_servers_per_switch):
    l = []
    for i in range(num_of_servers_per_switch):
        x = random_permutation_traffic_helper(G)
        l.append(x)
    ll = [item for sublist in l for item in sublist]
    lll = []
    for i in range(len(ll)):
        s, d = ll[i]
        lll.append((s, d, i))
    return lll


def all_to_all_helper(G):
    d = []
    for v in G.nodes():
        for v1 in G.nodes():
            if v != v1:
                d.append((v, v1))
    return d


def all_to_all(G, num_of_servers_per_switch):
    l = []
    for i in range(num_of_servers_per_switch):
        x = all_to_all_helper(G)
        l.append(x)
    ll = [item for sublist in l for item in sublist]
    lll = []
    for i in range(len(ll)):
        s, d = ll[i]
        lll.append((s, d, i))
    return lll


def av_number_of_hops_of_acc_con(paths_of_acc_c):
    l = [len(i) - 1 for i in paths_of_acc_c.values()]
    return sum(l), len(l)


def throughput_upper_bound(N, d, number_of_servers_per_rack):
    av_lb = lb_av_shortest_path(N, d)
    f = number_of_servers_per_rack * (N * (N - 1))
    th_up = (N * d) / (f * av_lb)
    return th_up


def sort_dict_by_value_lowest_to_highest(dic):
    return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}


def sort_paths_according_to_utilization_by_minimum(paths, ksp, utilization):
    path_utilization = dict()
    for path in paths:
        temp_path_edges_util = []
        # print(path)
        for edge in zip(path, path[1:]):
            # print(edge)
            # edge_name = (str(min(int(edge[0]), int(edge[1]))), str(max(int(edge[0]), int(edge[1]))))
            temp_edge_utilization = utilization.get(edge, 0)
            temp_path_edges_util.append(temp_edge_utilization)
        path_utilization[tuple(path)] = min(temp_path_edges_util)
    unique_hops = dict()
    for path in paths:
        unique_hops[len(path)] = dict()
    for path in paths:
        unique_hops[len(path)][tuple(path)] = path_utilization[tuple(path)]
    paths_to_be_returned = []
    for p, v in unique_hops.items():
        sorted_v = sort_dict_by_value_lowest_to_highest(v)
        for p1, v1 in sorted_v.items():
            paths_to_be_returned.append(list(p1))
    # return paths_to_be_returned[:ksp]
    return paths_to_be_returned


def sort_paths_according_to_utilization_by_maximum(paths, utilization):
    paths_to_be_returned = []
    for path in paths:
        temp_path_edges_util = []
        # print(path)
        for edge in zip(path, path[1:]):
            # print(edge)
            # edge_name = (str(min(int(edge[0]), int(edge[1]))), str(max(int(edge[0]), int(edge[1]))))
            temp_edge_utilization = utilization.get(edge, 0)
            temp_path_edges_util.append(temp_edge_utilization)
        # print(temp_path_edges_util)
        if max(temp_path_edges_util) < 1.0:
            paths_to_be_returned.append(path)
    return paths_to_be_returned

# def check_path_capacity(paths, utilization):


def find_ksp_throughput(G, ksp, traffic_matrix, alpha, util, fp=None, k=None):
    r = Routing(G)
    if fp is None:
        paths = r.ksp(k)
    else:
        paths = fp
    # print(paths)
    # print(len(list(G.edges())))
    utilization = util.copy()
    connections_on_edges = dict()
    accepted_connections = []
    blocked_connections = []
    paths_of_accepted_connections = dict()
    for s, d, id_ in traffic_matrix:
        paths_s = sort_paths_according_to_utilization_by_minimum(paths[s][d], ksp, utilization)
        counter = 0
        new_alpha = alpha
        for path in paths_s:
            counter += 1
            temp_path_edges_util = []
            temp_path_edges_util1 = []
            if new_alpha < 0.00001:
                break
            for edge_name in zip(path, path[1:]):
                # edge_name = (str(min(int(edge[0]), int(edge[1]))), str(max(int(edge[0]), int(edge[1]))))
                temp_edge_utilization = utilization.get(edge_name, 0) + new_alpha
                temp_path_edges_util1.append(utilization.get(edge_name, 0))
                temp_path_edges_util.append(temp_edge_utilization)
                if edge_name not in connections_on_edges:
                    connections_on_edges[edge_name] = []

            if max(temp_path_edges_util) <= 1.0:
                for edge_name in zip(path, path[1:]):
                    # edge_name = (str(min(int(e[0]), int(e[1]))), str(max(int(e[0]), int(e[1]))))
                    temp_edge_utilization = float("{:.2f}".format(
                        float("{:.2f}".format(utilization.get(edge_name, 0))) + float("{:.2f}".format(new_alpha))))
                    utilization[edge_name] = temp_edge_utilization
                    connections_on_edges[edge_name].append((s, d, id_))
                accepted_connections.append((s, d, id_))
                paths_of_accepted_connections[(s, d, id_)] = path
                break

            elif max(temp_path_edges_util1) < 1.0:
                capacity = 1.0 - max(temp_path_edges_util1)
                to_send = capacity
                for edge_name in zip(path, path[1:]):
                    # edge_name = (str(min(int(e[0]), int(e[1]))), str(max(int(e[0]), int(e[1]))))
                    temp_edge_utilization = float("{:.2f}".format(
                        float("{:.2f}".format(utilization.get(edge_name, 0))) + float("{:.2f}".format(to_send))))
                    utilization[edge_name] = temp_edge_utilization
                    if (s, d, id_) not in connections_on_edges[edge_name]:
                        connections_on_edges[edge_name].append((s, d, id_))
                new_alpha = new_alpha - to_send

            elif counter == len(paths_s):
                # print("#######################")
                # print("blocked connections")
                # alternative_paths = sort_paths_according_to_utilization_by_maximum(paths[s][d], utilization)
                # print(alternative_paths)
                # for p in alternative_paths:
                #     for edge_name in zip(p, p[1:]):
                #         print(utilization.get(edge_name, 0), " ")
                #     print('\n')
                # print("#######################")
                blocked_connections.append((s, d, id_))

    num_of_ac_c = len(accepted_connections)
    num_of_b_c = len(blocked_connections)
    return alpha, num_of_b_c, num_of_ac_c, accepted_connections, blocked_connections, traffic_matrix, paths_of_accepted_connections, utilization, connections_on_edges


strat_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d15/'
# xpander_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_xpander/d24/'
jellyfish_path = 'C:/Users/umroot/PycharmProjects/datacenter/new_jellyfish/d15/'
G = read_graph_from_file(strat_path + 'StratAL1024_8.txt')
# G1 = read_graph_from_file(xpander_path + 'XpanderAL2048_8.txt')
G2 = read_graph_from_file(jellyfish_path + 'JellyfishAL1024_8.txt')

write_paths_to_file(G, 5, 'C:/Users/umroot/PycharmProjects/datacenter/computed_k_paths/strat128_15_5_v3')
write_paths_to_file(G2, 5, 'C:/Users/umroot/PycharmProjects/datacenter/computed_k_paths/jellyfish128_15_5_v3')

strat_paths = read_paths_from_file('C:/Users/umroot/PycharmProjects/datacenter/computed_k_paths/strat128_15_5_v3.json')
# xpander_paths = read_paths_from_file('xpander20.json')
jellyfish_paths = read_paths_from_file('C:/Users/umroot/PycharmProjects/datacenter/computed_k_paths/jellyfish128_15_5_v3.json')

number_of_servers_under_each_switch = [1]
alpha_f = 0.0618
toposs = ['strat']  #'jellyfish'
topos = {'strat': G} #,'jellyfish':G2
paths_ = {'strat': strat_paths} #, 'jellyfish':jellyfish_paths


ks = [1]
util = dict()
l1 = []
l2 = []
l3 = []
for i in range(1000):
    for k in ks:
        topos_th = {'strat': {}, 'jellyfish': {}}
        topos_uth = {'strat': {}, 'jellyfish': {}}
        topos_acc = {'strat': {},  'jellyfish': {}}
        topos_bl = {'strat': {},  'jellyfish': {}}
        topos_av_num_of_hops = {'strat': {}, 'jellyfish': {}}
        for t in toposs:
            for j in number_of_servers_under_each_switch:
                t_m = all_to_all(topos[t], j)
                random.shuffle(t_m)
                alpha, num_of_b_c, num_of_ac_c, accepted_connections, blocked_connections, traffic_matrix, paths_of_accepted_connections, utilization, connections_on_edges = find_ksp_throughput(topos[t], k, t_m, alpha_f, util, fp=paths_[t])
                av_num_of_hops, con_num = av_number_of_hops_of_acc_con(paths_of_accepted_connections)
                topos_th[t][j] = "{:.3f}".format(alpha)
                topos_uth[t][j] = "{:.3f}".format(throughput_upper_bound(32, 15, 1))
                topos_bl[t][j] = "{:.3f}".format(num_of_b_c)
                topos_acc[t][j] = "{:.3f}".format(num_of_ac_c)
                topos_av_num_of_hops[t][j] = "{:.3f}".format(av_num_of_hops/con_num)
                l1.append(alpha)
                l2.append(num_of_ac_c)
                l3.append(num_of_b_c)
                # print(t, max(list(utilization.values())), min(list(utilization.values())))
                # print(t, len(list(utilization.keys())))
        # print("##################################################################")
        # print("k", k)
        # print(topos_th)
        # print(topos_uth)
        # print(topos_acc)
        # print(topos_bl)
        # print(topos_av_num_of_hops)
        # print("##################################################################")
print(np.mean(l1))
print(np.mean(l2))
print(np.mean(l3))
