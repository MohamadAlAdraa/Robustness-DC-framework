import random
import math
import time
import networkx as nx

from routing import Routing
import numpy as np
import pulp as pl

from utilities import read_graph_from_file, plot_path_lengths, lb_av_shortest_path


def random_permutation_traffic_helper(G, demand):
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
        traffic_matric_random_one_to_one.append((nodes[i], temp_nodes[i], demand))
    return traffic_matric_random_one_to_one


def random_permutation_traffic(G, demand, num_of_servers_per_switch):
    l = []
    for i in range(num_of_servers_per_switch):
        x = random_permutation_traffic_helper(G, demand)
        l.append(x)
    ll =  [item for sublist in l for item in sublist]
    lll = []
    for i in range(len(ll)):
        s,d,r = ll[i]
        lll.append((s,d,r,i))
    return lll


def all_to_all_helper(G, demand):
    d = []
    for v in G.nodes():
        for v1 in G.nodes():
            if v != v1:
                d.append((v, v1, demand))
    return d


def all_to_all(G, demand, num_of_servers_per_switch):
    l = []
    for i in range(num_of_servers_per_switch):
        x = all_to_all_helper(G, demand)
        l.append(x)
    ll = [item for sublist in l for item in sublist]
    lll = []
    for i in range(len(ll)):
        s, d, r = ll[i]
        lll.append((s, d, r, i))
    return lll


def prepare_variables(G, traffic_matrix, k):
    g_directed = G.to_directed()
    r = Routing(g_directed)
    k_paths = r.ksp(k)
    # print(k_paths)
    edges = sorted(list(g_directed.edges()).copy())
    C = {e: 0 for e in edges}
    flows_on_edges_ = dict()
    edges_on_flows_ = dict()
    flows_on_nodes_ = dict()
    for s, d, r, id_ in traffic_matrix:
        flows_on_edges_[(s, d, r, id_)] = []
        flows_on_nodes_[(s, d, r, id_)] = []
        for path in k_paths[s][d]:
            for edge_name in zip(path, path[1:]):
                if edge_name[0] not in flows_on_nodes_[(s, d, r, id_)]:
                    flows_on_nodes_[(s, d, r, id_)].append(edge_name[0])
                if edge_name[1] not in flows_on_nodes_[(s, d, r, id_)]:
                    flows_on_nodes_[(s, d, r, id_)].append(edge_name[1])
                if edge_name not in list(edges_on_flows_.keys()):
                    edges_on_flows_[edge_name] = []
                if edge_name not in flows_on_edges_[(s, d, r, id_)]:
                    flows_on_edges_[(s, d, r, id_)].append(edge_name)
                if (s, d, r, id_) not in edges_on_flows_[edge_name]:
                    edges_on_flows_[edge_name].append((s, d, r, id_))
                C[edge_name] = 1.0
    return flows_on_edges_, edges_on_flows_, flows_on_nodes_, C


def optimal_routing(G_, flows_on_edges_, edges_on_flows_, flows_on_nodes_, C_, toponame, k):
    topo = G_.to_directed()
    # nodes = sorted(list(topo.nodes()).copy())
    # edges = sorted(list(topo.edges()).copy())
    C = C_.copy()

    # DEFINE THE PROBLEM
    mcf_model = pl.LpProblem('MCF', pl.LpMaximize)

    # DEFINE THE VARIABLES
    alpha = pl.LpVariable('alpha', lowBound=0, upBound=1, cat=pl.LpContinuous) # alpha variable
    flow_edges = dict() # flow variable for each edge
    for f, l in flows_on_edges_.items():
        for e in l:
            variable_name = 'f{}_{}_{}_{}_{}_{}'.format(e[0], e[1], f[0], f[1], f[2], f[3])
            flow_edges[(e, f)] = pl.LpVariable(variable_name, lowBound=0, upBound=f[2], cat=pl.LpContinuous)

    # DEFINE THE CONSTRAINTS
    for e, F in edges_on_flows_.items(): # capacity constraint
        Cs = []
        for f in F:
            Cs.append(flow_edges[(e, f)])
        mcf_model += pl.lpSum(Cs) <= C[e]

    for f in flows_on_edges_.keys():
        for v in flows_on_nodes_[f]:
            in_edges = topo.in_edges(v)
            out_edges = topo.out_edges(v)
            Cs = []
            Cs1 = []
            for e in in_edges:
                if e in flows_on_edges_[f]:
                    Cs.append(flow_edges[(e, f)])
            for e in out_edges:
                if e in flows_on_edges_[f]:
                    Cs1.append(flow_edges[(e, f)])
            if v == f[0]:
                mcf_model += ((pl.lpSum(Cs1)) - (pl.lpSum(Cs))) == alpha * f[2]
            elif v == f[1]:
                mcf_model += ((pl.lpSum(Cs1)) - (pl.lpSum(Cs))) == -1.0 * alpha * f[2]
            else:
                mcf_model += pl.lpSum(Cs) == pl.lpSum(Cs1)

    # DEFINE THE OBJECTIVE
    mcf_model += alpha

    f = open("C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/times/" + toponame + str(k) + ".txt", "a")
    start = time.process_time()
    # calling the solver
    cplex = pl.get_solver('CPLEX_CMD')
    mcf_model.solve(solver=cplex)
    f.write(str(time.process_time() - start))
    # write the result
    # with open(path + topo_name + topo_size, 'w') as f:
    #     r = 'alpha: ' + str(mcf_model.objective.value()) +'\n' + 'status: ' + str(pl.LpStatus[mcf_model.status])
    #     f.write(r)
    d = dict()
    for i in mcf_model.variables():
        if i.value() >= 0.0000001:
            d[str(i)] = i.value()
    paths = dict()
    for f_, l in flows_on_edges_.items():
        temp_p = []
        for e in l:
            s = 'f{}_{}_{}_{}_{}_{}'.format(e[0], e[1], f_[0], f_[1], f_[2], f_[3])
            if s in d:
                temp_p.append(e)
        paths[f_] = temp_p
    # utilization = {e: 0.0 for e in edges}
    # flows_on_edges = {e: [] for e in edges}
    # for e in edges:
    #     for f in F:
    #         s = 'f{}_{}_{}_{}_{}_{}'.format(e[0], e[1], f[0], f[1], f[2], f[3])
    #         if s in d:
    #             utilization[e] = utilization.get(e) + d[s]
    #             flows_on_edges[e].append(f)
    # for the rerout problem I may need to return the following attributes: topo, mcf_model.objective.value(), d, paths, utilization, flows_on_edges
    return mcf_model.objective.value(), paths


def throughput_upper_bound(N, d, number_of_servers_per_rack):
    av_lb = lb_av_shortest_path(N, d)
    f = number_of_servers_per_rack*(N*(N-1))
    th_up = (N*d) / (f*av_lb)
    return th_up


def check_degree(G):
    d = list(G.degree())
    dd = []
    for i in d:
        dd.append(i[1])
    print('Max degree', max(dd))
    print('Min degree', min(dd))


def avsp(paths_of_acc_c):
    l = [len(i) for i in paths_of_acc_c.values()]
    return sum(l)/len(l)
# G = nx.random_regular_graph(3, 8)
# t = all_to_all(G, 1.0, 1)
# k = 10
# flows_on_edges, edges_on_flows, flows_on_nodes, C = prepare_variables(G, t, k)
# alpha = optimal_routing(G, flows_on_edges, edges_on_flows, flows_on_nodes, C)
# print(alpha)


# strat_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d15/'
# jellyfish_path = 'C:/Users/umroot/PycharmProjects/datacenter/new_jellyfish/d15/'
# G = read_graph_from_file(strat_path + 'StratAL1024_8.txt')
# G1 = read_graph_from_file(jellyfish_path + 'JellyfishAL1024_8.txt')
# k = 10
# l = []
# l1 = []
# l2 = []
# l3 = []
# x = None
# x1 = None
# for i in range(1):
#     t = all_to_all(G, 1.0, 1)
#     t1 = all_to_all(G1, 1.0, 1)
#     random.shuffle(t)
#     random.shuffle(t1)
#     x = t
#     x1 = t1
#     flows_on_edges, edges_on_flows, flows_on_nodes, C = prepare_variables(G, t, k)
#     flows_on_edges1, edges_on_flows1, flows_on_nodes1, C1 = prepare_variables(G1, t1, k)
#     alpha, paths = optimal_routing(G, flows_on_edges, edges_on_flows, flows_on_nodes, C)
#     alpha1, paths1 = optimal_routing(G1, flows_on_edges1, edges_on_flows1, flows_on_nodes1, C1)
#     l.append(alpha)
#     l1.append(alpha1)
#     l2.append(paths)
#     l3.append(paths1)
# print('Strat', np.mean(l), len(x))
# print('Jellyfish', np.mean(l1), len(x1))
# print('Strat', avsp(l2[0]))
# print('Jellyfish', avsp(l3[0]))
# print('Upper bound', throughput_upper_bound(len(list(G.nodes())), 15, 1))

path = 'C:/Users/umroot/PycharmProjects/datacenter/topo_for_optimal_routing/topo_used/'
path1 = 'C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/'

params = [(128, 1024, 23, 8)] #(32, 256, 23, 8), (64, 512, 23, 8), (128, 1024, 23, 8), (256, 2048, 23, 8), (512, 4096, 23, 8)


# Max-Min flow problem
ks = [1,2,3,4,5,10,20,30,40] #10, 15, 30
# d = {"strat": {}, "xpander": {}, "jellyfish": {}}
# d1 = {"strat": {}, "xpander": {}, "jellyfish": {}}
# d2 = {"strat": {}, "xpander": {}, "jellyfish": {}}
for k in ks:
    f = open(path1 + "result_max_min_flow_num_s.txt", "a")
    f.write('k = ' + str(k) + ':' + '\n')
    f.close()
    for p in params:
        f = open(path1 + "result_max_min_flow_num_s.txt", "a")
        num_of_switches, num_servers, switch_k, num_servers_per_rack = p
        switch_d = switch_k - num_servers_per_rack
        strat = []
        xpander = []
        jellyfish = []
        strat_ = []
        xpander_ = []
        jellyfish_ = []
        th_up = None
        for i in range(1, 2):
            strat_file = "StratAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
            strat_graph = read_graph_from_file(path + strat_file)
            # xpander_file = "XpanderAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
            # xpander_graph = read_graph_from_file(path + xpander_file)
            # jellyfish_file = "JellyfishAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
            # jellyfish_graph = read_graph_from_file(path + jellyfish_file)
            F = all_to_all(strat_graph, 1.0, 1)
            # random.shuffle(F)
            flows_on_edges, edges_on_flows, flows_on_nodes, C = prepare_variables(strat_graph, F, k)
            # flows_on_edges1, edges_on_flows1, flows_on_nodes1, C1 = prepare_variables(xpander_graph, F, k)
            # flows_on_edges2, edges_on_flows2, flows_on_nodes2, C2 = prepare_variables(jellyfish_graph, F, k)
            strat_alpha, strat_paths = optimal_routing(strat_graph, flows_on_edges, edges_on_flows, flows_on_nodes, C, "Strat", k)
            # xpander_alpha, xpander_paths = optimal_routing(xpander_graph, flows_on_edges1, edges_on_flows1, flows_on_nodes1, C1)
            # jellyfish_alpha, jellyfish_paths = optimal_routing(jellyfish_graph, flows_on_edges2, edges_on_flows2, flows_on_nodes2, C2)
            th_up = throughput_upper_bound(num_of_switches, switch_d, 1)
            # strat.append(strat_alpha)
            # xpander.append(xpander_alpha)
            # jellyfish.append(jellyfish_alpha)
            # strat_.append(avsp(strat_paths))
            # print(strat_paths)
            # xpander_.append(avsp(xpander_paths))
            # jellyfish_.append(avsp(jellyfish_paths))
        # f.write(str(num_of_switches) + ' ' + str(strat[0]) + ' ' + str(xpander[0]) + ' ' + str(jellyfish[0]) + ' ' + str(th_up) + ' ' + str(strat[0]/th_up) + ' ' + str(xpander[0]/th_up) + ' ' + str(jellyfish[0]/th_up) + ' ' + str(strat_[0]) + ' ' + str(xpander_[0]) + ' ' + str(jellyfish_[0]) + '\n')
        # f.write(str(num_of_switches) + ' ' + str(np.mean(strat)) + ' ' + str(np.mean(xpander)) + ' ' + str(np.mean(jellyfish)) + ' ' + str(th_up) + ' ' + str(np.mean(strat_)) + ' ' + str(np.mean(xpander_)) + ' ' + str(np.mean(jellyfish_)) + '\n')
        # f.write(str(num_of_switches) + ' ' + str(np.mean(strat)) + ' ' + str(np.mean(jellyfish)) + ' ' + str(th_up) + ' ' + str(np.mean(strat_)) + ' ' + str(np.mean(jellyfish_)) + '\n')
        f.close()
# plot_path_lengths(d, "C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/max_min_flow_num_s", "Number Of Servers", "Normalized Throughput")
# plot_path_lengths(d1, "C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/max_min_flow_num_s_norm", "Number Of Servers", "Normalized Throughput")
# plot_path_lengths(d2, "C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/max_min_flow_num_s_norm_per", "Number Of Servers", "Normalized Throughput (%)")

