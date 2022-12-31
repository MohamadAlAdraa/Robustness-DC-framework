import random
import numpy as np
from routing import Routing
from utilities import read_graph_from_file, lb_av_shortest_path
import json


# def all_to_all(G):
#     traffic_matrix_all_to_all = []
#     for i in G.nodes():
#         for j in G.nodes():
#             if i != j:
#                 traffic_matrix_all_to_all.append((i, j))
#     return traffic_matrix_all_to_all
#
#
# def random_permutation(matrix):
#     random.shuffle(matrix)
#     return matrix
#
#
# def random_permutation_traffic(G):
#     traffic_matric_random_one_to_one = []
#     nodes = list(G.nodes())
#     b= True
#     while b:
#         temp_nodes = list(G.nodes())
#         random.shuffle(temp_nodes)
#         b = False
#         for i in range(len(nodes)):
#             if nodes[i] == temp_nodes[i]:
#                 b = True
#                 break
#     for i in range(len(nodes)):
#         traffic_matric_random_one_to_one.append((nodes[i], temp_nodes[i]))
#     return traffic_matric_random_one_to_one


def sort_dict_by_value_lowest_to_highest(dict):
    return {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}


def sort_paths_according_to_utilization(paths, ksp_, utilization):
    path_utilization = dict()
    for path in paths:
        temp_path_edges_util = []
        for edge in zip(path, path[1:]):
            edge_name = (str(min(int(edge[0]), int(edge[1]))), str(max(int(edge[0]), int(edge[1]))))
            temp_edge_utilization = utilization.get(edge_name, 0)
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
    return paths_to_be_returned[:ksp_]


def find_ksp_throughput(G, ksp_, traffic_matrix, alpha, util, fp=None, k=None):
    r = Routing(G)
    if fp==None:
        paths = r.ksp(k)
    else:
        paths = fp
    utilization = util.copy()
    connetions_on_edges = dict()
    accepted_connections = []
    blocked_connections = []
    paths_of_accepted_connections = dict()
    for s, d, id in traffic_matrix:
        paths_s = sort_paths_according_to_utilization(paths[s][d], ksp_, utilization)
        counter = 0
        for path in paths_s:
            counter += 1
            temp_path_edges_util = []
            for edge in zip(path, path[1:]):
                edge_name = (str(min(int(edge[0]), int(edge[1]))), str(max(int(edge[0]), int(edge[1]))))
                temp_edge_utilization = utilization.get(edge_name, 0) + alpha
                temp_path_edges_util.append(temp_edge_utilization)
                if edge_name not in connetions_on_edges:
                    connetions_on_edges[edge_name] = []
            if max(temp_path_edges_util) <= 2.0:
                for e in zip(path, path[1:]):
                    edge_name = (str(min(int(e[0]), int(e[1]))), str(max(int(e[0]), int(e[1]))))
                    temp_edge_utilization = float("{:.2f}".format(float("{:.2f}".format(utilization.get(edge_name, 0))) + float("{:.2f}".format(alpha))))
                    utilization[edge_name] = temp_edge_utilization
                    connetions_on_edges[edge_name].append((s,d, id))
                accepted_connections.append((s,d, id))
                paths_of_accepted_connections[(s, d, id)] = path
                break
            elif counter == len(paths_s):
                blocked_connections.append((s, d, id))

    num_of_ac_c = len(accepted_connections)
    num_of_b_c= len(blocked_connections)
    return alpha, num_of_b_c, num_of_ac_c, accepted_connections, blocked_connections, traffic_matrix, paths_of_accepted_connections, utilization, connetions_on_edges


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
    traffic_matric_random_one_to_one = []
    nodes = list(G.nodes())
    b= True
    while b:
        temp_nodes = list(G.nodes())
        random.shuffle(temp_nodes)
        b = False
        for i in range(len(nodes)):
            if nodes[i] == temp_nodes[i]:
                b = True
                break
    for i in range(len(nodes)):
        traffic_matric_random_one_to_one.append((nodes[i], temp_nodes[i]))
    return traffic_matric_random_one_to_one


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


# def tt(num_of_servers_per_switch, G):
#     l = []
#     for i in range(num_of_servers_per_switch):
#         x = random_permutation_traffic(G)
#         l.append(x)
#     ll = [item for sublist in l for item in sublist]
#     lll = []
#     for i in range(len(ll)):
#         s,d = ll[i]
#         lll.append((s,d,i))
#     return lll


def av_number_of_hops_of_acc_con(paths_of_acc_c):
    l = [len(i) - 1 for i in paths_of_acc_c.values()]
    return sum(l), len(l)


def throughput_upper_bound(N, d, number_of_servers_per_rack):
    av_lb = lb_av_shortest_path(N, d)
    f = number_of_servers_per_rack*(N*(N-1))
    th_up = (N*d) / (f*av_lb)
    return th_up


strat_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d15/'
# xpander_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_xpander/d24/'
jellyfish_path = 'C:/Users/umroot/PycharmProjects/datacenter/new_jellyfish/d15/'
G = read_graph_from_file(strat_path + 'StratAL1024_8.txt')
# G1 = read_graph_from_file(xpander_path + 'XpanderAL2048_8.txt')
G2 = read_graph_from_file(jellyfish_path + 'JellyfishAL1024_8.txt')

# write_paths_to_file(G, 15, 'C:/Users/umroot/PycharmProjects/datacenter/computed_k_paths/strat15')
# write_paths_to_file(G2, 15, 'C:/Users/umroot/PycharmProjects/datacenter/computed_k_paths/jellyfish15')


strat_paths15 = read_paths_from_file('C:/Users/umroot/PycharmProjects/datacenter/computed_k_paths/strat15.json')
# xpander_paths20 = read_paths_from_file('xpander20.json')
jellyfish_paths15 = read_paths_from_file('C:/Users/umroot/PycharmProjects/datacenter/computed_k_paths/jellyfish15.json')


number_of_servers_under_each_switch = [1]
alpha_f = 0.049
toposs = ['strat', 'jellyfish']
topos = {'strat': G, 'jellyfish':G2}
paths_ = {'strat': strat_paths15, 'jellyfish':jellyfish_paths15}


ks = [10]
util = dict()
for k in ks:
    topos_th = {'strat': {}, 'jellyfish': {}}
    topos_uth = {'strat': {}, 'jellyfish': {}}
    topos_acc = {'strat': {},  'jellyfish': {}}
    topos_bl = {'strat': {},  'jellyfish': {}}
    topos_av_num_of_hops = {'strat': {}, 'jellyfish': {}}
    for t in toposs:
        for j in number_of_servers_under_each_switch:
            # s = []
            # b = []
            # c1 = []
            # c2 = []
            # a = 0
            print(j)
            alpha, num_of_b_c, num_of_ac_c, accepted_connections, blocked_connections, traffic_matrix, paths_of_accepted_connections, utilization, connetions_on_edges = find_ksp_throughput(topos[t], k, all_to_all(topos[t],j), alpha_f, util, fp=paths_[t])
            # s.append(num_of_ac_c)
            # b.append(num_of_b_c)
            av_num_of_hops, con_num = av_number_of_hops_of_acc_con(paths_of_accepted_connections)
            # c1.append(av_num_of_hops)
            # c2.append(con_num)
            # a= alpha
            topos_th[t][j] = "{:.3f}".format(alpha)
            topos_uth[t][j] = "{:.3f}".format(throughput_upper_bound(128, 15, 1))
            topos_bl[t][j] = "{:.3f}".format(num_of_b_c)
            topos_acc[t][j] = "{:.3f}".format(num_of_ac_c)
            topos_av_num_of_hops[t][j] = "{:.3f}".format(av_num_of_hops/con_num)

    print(topos_th)
    print(topos_uth)
    print(topos_acc)
    print(topos_bl)
    print(topos_av_num_of_hops)
#     with open("topos_alphas_without_distribution_throughput_k_" + str(k) + '.json', 'w') as f:
#         json.dump(topos_th, f)
#     with open("topos_alphas_without_distribution_blocked_k_" + str(k) + '.json', 'w') as f:
#         json.dump(topos_bl, f)
#     with open("topos_alphas_without_distribution_acc_k_" + str(k) + '.json', 'w') as f:
#         json.dump(topos_acc, f)
#     with open("topos_alphas_without_distribution_av_num_of_hops_k_" + str(k) + '.json', 'w') as f:
#         json.dump(topos_av_num_of_hops, f)



