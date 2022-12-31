import json
import random
import numpy as np
from routing import Routing
from utilities import read_graph_from_file, lb_av_shortest_path


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


def read_paths_from_file(filename):
    f = open(filename)
    data = json.load(f)
    return data


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


def sort_dict_by_value_lowest_to_highest(dict):
    return {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}


def distribute_traffic_evenly(utilization, paths, alpha):
    temp_paths = paths.copy()
    temp_utilization = utilization.copy()
    b = True
    while(b):
        temp_utilization1 = temp_utilization.copy()
        l = len(temp_paths)
        b1 = True
        for path in temp_paths:
            temp_path_edges_util = []
            for edge in zip(path, path[1:]):
                edge_name = (str(min(int(edge[0]), int(edge[1]))), str(max(int(edge[0]), int(edge[1]))))
                temp_edge_utilization = temp_utilization1.get(edge_name, 0) + (alpha/l)
                temp_path_edges_util.append(temp_edge_utilization)
            if max(temp_path_edges_util) > 1.0:
                temp_paths.remove(path)
                b1 = False
                break
        if b1:
            b = False

    return temp_paths


def get_blocked_connections(acc, all_c):
    acc_c = []
    for i, j in acc.items():
        for k in j:
            acc_c.append((i, k))

    return [i for i in all_c if i not in acc_c]


def find_ksp_throughput_evenly_distributed(G, ksp_, traffic_matrix, alpha, fp=None, k=None):
    r = Routing(G)
    if fp==None:
        paths = r.ksp(k)
    else:
        paths = fp
    utilization = dict()
    accepted_connections = dict()
    paths_of_accepted_connections = dict()
    for g in G.nodes():
        accepted_connections[g] = []
    for s, d, id in traffic_matrix:
        paths_s = sort_paths_according_to_utilization(paths[s][d], ksp_, utilization)
        paths_ss = distribute_traffic_evenly(utilization, paths_s, alpha)
        if len(paths_ss) != 0:
            for path in paths_ss:
                for edge in zip(path, path[1:]):
                    edge_name = (str(min(int(edge[0]), int(edge[1]))), str(max(int(edge[0]), int(edge[1]))))
                    temp_edge_utilization = utilization.get(edge_name, 0) + alpha/len(paths_ss)
                    utilization[edge_name] = temp_edge_utilization
            paths_of_accepted_connections[(s, d, id)] = paths_ss
            accepted_connections[s].append(d)
    num_of_ac_c = 0
    for i in accepted_connections.values():
        num_of_ac_c += len(i)
    total_c = len(traffic_matrix)
    num_of_b_c = total_c - num_of_ac_c
    return alpha, num_of_b_c, num_of_ac_c, accepted_connections, traffic_matrix, paths_of_accepted_connections, utilization


# def tt(num_of_servers_per_switch, G):
#     l = []
#     for i in range(num_of_servers_per_switch):
#         x = random_permutation_traffic(G)
#         l.append(x)
#     return [item for sublist in l for item in sublist]


def avsp_of_acc_con(paths_of_acc_c):
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
alpha_f = 0.03
toposs = ['strat', 'jellyfish']
topos = {'strat': G, 'jellyfish':G2}
paths_ = {'strat': strat_paths15, 'jellyfish':jellyfish_paths15}


ks = [2]
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
            alpha, num_of_b_c, num_of_ac_c, accepted_connections, traffic_matrix, paths_of_accepted_connections, utilization = find_ksp_throughput_evenly_distributed(topos[t], k, all_to_all(topos[t],j), alpha_f, fp=paths_[t])
            # s.append(num_of_ac_c)
            # b.append(num_of_b_c)
            av_num_of_hops, con_num = avsp_of_acc_con(paths_of_accepted_connections)
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



# ks = [6]
# for k in ks:
#     topos_th = {'strat': {}, 'xpander': {}, 'jellyfish': {}}
#     topos_acc = {'strat': {}, 'xpander': {}, 'jellyfish': {}}
#     topos_bl = {'strat': {}, 'xpander': {}, 'jellyfish': {}}
#     topos_avsp = {'strat': {}, 'xpander': {}, 'jellyfish': {}}
#     for t in toposs:
#         for j in alphas:
#             s = []
#             b = []
#             c1 = []
#             c2 = []
#             a = 0
#             print(j)
#             for i in range(10):
#                 print(i, end=',')
#                 alpha, num_of_b_c, num_of_ac_c, accepted_connections, traffic_matrix, paths_of_accepted_connections, utilization = find_ksp_throughput(topos[t], k, tt(j*8, topos[t]), alpha_f, fp=paths_[t])
#                 s.append(num_of_ac_c)
#                 b.append(num_of_b_c)
#                 avsp, con_num = avsp_of_acc_con(paths_of_accepted_connections)
#                 c1.append(avsp)
#                 c2.append(con_num)
#                 a= alpha
#             topos_th[t][j] = "{:.3f}".format(np.mean(s)*a)
#             topos_bl[t][j] = "{:.3f}".format((np.mean(b)))
#             topos_acc[t][j] = "{:.3f}".format((np.mean(s)))
#             topos_avsp[t][j] = "{:.3f}".format(sum(c1)/sum(c2))
#
#     with open("topos_alphas_with_distribution_throughput_k_" + str(k) + '.json', 'w') as f:
#         json.dump(topos_th, f)
#     with open("topos_alphas_with_distribution_blocked_k_" + str(k) + '.json', 'w') as f:
#         json.dump(topos_bl, f)
#     with open("topos_alphas_with_distribution_acc_k_" + str(k) + '.json', 'w') as f:
#         json.dump(topos_acc, f)
#     with open("topos_alphas_with_distribution_avsp_k_" + str(k) + '.json', 'w') as f:
#         json.dump(topos_avsp, f)