import random
import math
import time
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


def optimal_routing(G, tf, toponame):
    topo = G.to_directed()
    nodes = sorted(list(topo.nodes()).copy())
    edges = sorted(list(topo.edges()).copy())
    F = tf.copy()
    C = {e: 1.0 for e in edges}

    # DEFINE THE PROBLEM
    mcf_model = pl.LpProblem('MCF', pl.LpMaximize)

    # DEFINE THE VARIABLES
    alpha = pl.LpVariable('alpha', lowBound=0, upBound=1, cat=pl.LpContinuous) # alpha variable
    flow_edges = dict() # flow variable for each edge
    for e in edges:
        for f in F:
            variable_name = 'f{}_{}_{}_{}_{}_{}'.format(e[0], e[1], f[0], f[1], f[2], f[3])
            flow_edges[(e,f)] = pl.LpVariable(variable_name, lowBound=0, upBound=1, cat=pl.LpContinuous)

    # DEFINE THE CONSTRAINTS
    for e in edges: # capacity constraint
        Cs = []
        for f in F:
            Cs.append(flow_edges[(e,f)])
        mcf_model += pl.lpSum(Cs) <= C[e]

    for v in nodes: # flow conservation constraint
        for f in F:
            in_edges = topo.in_edges(v)
            out_edges = topo.out_edges(v)
            Cs = []
            Cs1 = []
            for e in in_edges:
                Cs.append(flow_edges[(e,f)])
            for e in out_edges:
                Cs1.append(flow_edges[(e,f)])
            if v == f[0]:
                mcf_model += ((pl.lpSum(Cs1)) - (pl.lpSum(Cs))) == alpha
            elif v == f[1]:
                mcf_model += ((pl.lpSum(Cs1)) - (pl.lpSum(Cs))) == -1.0 * alpha
            else:
                mcf_model += pl.lpSum(Cs) == pl.lpSum(Cs1)

    # DEFINE THE OBJECTIVE
    mcf_model += alpha

    # calling the solver
    f = open("C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/" + toponame + ".txt", "a")
    start = time.process_time()
    # your code here
    cplex = pl.get_solver('CPLEX_CMD')
    mcf_model.solve(solver=cplex)
    f.write(str(time.process_time() - start))
    # write the result
    # with open(path + topo_name + topo_size, 'w') as f:
    #     r = 'alpha: ' + str(mcf_model.objective.value()) +'\n' + 'status: ' + str(pl.LpStatus[mcf_model.status])
    #     f.write(r)
    d = dict()
    for i in mcf_model.variables():
        if i.value() != 0.0:
            d[str(i)] = i.value()
    paths = dict()
    for f in F:
        temp_p = []
        for e in edges:
            s = 'f{}_{}_{}_{}_{}_{}'.format(e[0], e[1], f[0], f[1], f[2], f[3])
            if s in d:
                temp_p.append(e)
        paths[f] = temp_p
    utilization = {e: 0.0 for e in edges}
    flows_on_edges = {e: [] for e in edges}
    for e in edges:
        for f in F:
            s = 'f{}_{}_{}_{}_{}_{}'.format(e[0], e[1], f[0], f[1], f[2], f[3])
            if s in d:
                utilization[e] = utilization.get(e) + d[s]
                flows_on_edges[e].append(f)
    # for the rerout problem I may need to return the following attributes: topo, mcf_model.objective.value(), d, paths, utilization, flows_on_edges
    return mcf_model.objective.value(), paths


def maximizing_flow(G, tf):
    topo = G.to_directed()
    nodes = sorted(list(topo.nodes()).copy())
    edges = sorted(list(topo.edges()).copy())
    F = tf.copy()

    # X-axis represents the load on the network (I can choose all-to-all and then sample from it load%)
    # F1 = sorted(all_to_all(G, demand)).copy()
    # random.shuffle(F1)
    # number_of_flows_used = int(len(F1)*load)
    # F = random.sample(F1, number_of_flows_used)

    C = {e: 1.0 for e in edges}

    # DEFINE THE PROBLEM
    mcf_model = pl.LpProblem('MCF', pl.LpMaximize)

    # DEFINE THE VARIABLES
    alphas = {}
    for f in F:
        alpha_name = 'alpha_' + str(f[0]) + "_" + str(f[1]) + "_" + str(f[2]) + "_" + str(f[3])
        alphas[f] = pl.LpVariable(alpha_name, lowBound=0, upBound=f[2], cat=pl.LpContinuous)
    flow_edges = dict()  # flow variable for each edge
    for e in edges:
        for f in F:
            variable_name = 'f{}_{}_{}_{}_{}_{}'.format(e[0], e[1], f[0], f[1], f[2], f[3])
            flow_edges[(e, f)] = pl.LpVariable(variable_name, lowBound=0, upBound=f[2], cat=pl.LpContinuous)

    # DEFINE THE CONSTRAINTS
    for e in edges:  # capacity constraint
        Cs = []
        for f in F:
            Cs.append(flow_edges[(e, f)])
        mcf_model += pl.lpSum(Cs) <= C[e]

    for v in nodes:  # flow conservation constraint
        for f in F:
            in_edges = topo.in_edges(v)
            out_edges = topo.out_edges(v)
            Cs = []
            Cs1 = []
            for e in in_edges:
                Cs.append(flow_edges[(e, f)])
            for e in out_edges:
                Cs1.append(flow_edges[(e, f)])
            if v == f[0]:
                mcf_model += ((pl.lpSum(Cs1)) - (pl.lpSum(Cs))) == alphas[f] * f[2]
            elif v == f[1]:
                mcf_model += ((pl.lpSum(Cs1)) - (pl.lpSum(Cs))) == -1.0 * alphas[f] * f[2]
            else:
                mcf_model += pl.lpSum(Cs) == pl.lpSum(Cs1)

    # DEFINE THE OBJECTIVE
    mcf_model += pl.lpSum(alphas.values())

    # calling the solver
    cplex = pl.get_solver('CPLEX_CMD')
    mcf_model.solve(solver=cplex)
    return mcf_model.objective.value()


def maximizing_rerouted_admitted_flows(topo, util, demand):
    nodes = sorted(list(topo.nodes()).copy())
    edges = sorted(list(util.keys()).copy())
    F = demand
    C = util.copy()

    # DEFINE THE PROBLEM
    mcf_model = pl.LpProblem('MCF', pl.LpMaximize)

    # DEFINE THE VARIABLES
    alphas = pl.LpVariable.dicts('alpha', {f for f in F}, cat=pl.LpBinary)  # alpha variable
    flow_edges = dict()  # flow variable for each edge
    for e in edges:
        for f in F:
            variable_name = 'f{}_{}_{}_{}_{}_{}'.format(e[0], e[1], f[0], f[1], f[2], f[3])
            flow_edges[(e, f)] = pl.LpVariable(variable_name, lowBound=0, upBound=f[2], cat=pl.LpContinuous)
    # DEFINE THE CONSTRAINTS
    for e in edges:  # capacity constraint
        Cs = []
        for f in F:
            Cs.append(flow_edges[(e, f)])
        mcf_model += pl.lpSum(Cs) <= C[e]

    for v in nodes:  # flow conservation constraint
        for f in F:
            in_edges = topo.in_edges(v)
            out_edges = topo.out_edges(v)
            Cs = []
            Cs1 = []
            for e in in_edges:
                Cs.append(flow_edges[(e, f)])
            for e in out_edges:
                Cs1.append(flow_edges[(e, f)])
            if v == f[0]:
                mcf_model += ((pl.lpSum(Cs1)) - (pl.lpSum(Cs))) == alphas[f] * f[2]
            elif v == f[1]:
                mcf_model += ((pl.lpSum(Cs1)) - (pl.lpSum(Cs))) == -1.0 * alphas[f] * f[2]
            else:
                mcf_model += pl.lpSum(Cs) == pl.lpSum(Cs1)

    # DEFINE THE OBJECTIVE
    mcf_model += pl.lpSum(alphas.values())

    # calling the solver

    cplex = pl.get_solver('CPLEX_CMD')
    mcf_model.solve(solver=cplex)
    return mcf_model.objective.value()


def throughput_upper_bound(N, d, number_of_servers_per_rack):
    av_lb = lb_av_shortest_path(N, d)
    f = number_of_servers_per_rack*(N*(N-1))
    th_up = (N*d) / (f*av_lb)
    return th_up


def failure(G, f_r):
    number_of_edges_to_be_removed = math.floor(f_r * len(list(G.edges())))
    G_temp = G.copy()
    edges_to_be_removed = random.sample(G.edges(), k=number_of_edges_to_be_removed)
    edges_to_be_removed_after_failure = edges_to_be_removed.copy()
    for rm in edges_to_be_removed:
        G_temp.remove_edge(rm[0], rm[1])
    G_after_failure = G_temp.copy()
    return edges_to_be_removed_after_failure, G_after_failure

def avsp(paths_of_acc_c):
    l = [len(i) for i in paths_of_acc_c.values()]
    return sum(l)/len(l)

path = 'C:/Users/umroot/PycharmProjects/datacenter/topo_for_optimal_routing/topo_used/'
path1 = 'C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/'

params = [(128, 1024, 23, 8)]


# Max-Min flow problem
d = {"strat":{}, "xpander":{}, "jellyfish":{}}
d1 = {"strat":{}, "xpander":{}, "jellyfish":{}}
d2 = {"strat":{}, "xpander":{}, "jellyfish":{}}

for p in params:
    print(p)
    for i in range(1, 2):
        num_of_switches, num_servers, switch_k, num_servers_per_rack = p
        switch_d = switch_k - num_servers_per_rack
        strat_file = "StratAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
        strat_graph = read_graph_from_file(path + strat_file)
        # xpander_file = "XpanderAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
        # xpander_graph = read_graph_from_file(path + xpander_file)
        # jellyfish_file = "JellyfishAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
        # jellyfish_graph = read_graph_from_file(path + jellyfish_file)
        F = sorted(all_to_all(strat_graph, 1.0, i)).copy()
        strat_alpha, strat_paths = optimal_routing(strat_graph, F, 'strat')
        print(strat_paths)
        # xpander_alpha, xpander_paths = optimal_routing(xpander_graph, F, 'xpander')
        # print(xpander_paths)
        # jellyfish_alpha, jellyfish_paths = optimal_routing(jellyfish_graph, F, 'jellyfish')
        # print(jellyfish_paths)
        th_up = throughput_upper_bound(num_of_switches, switch_d, i)
        d["strat"][int(i*num_of_switches)] = strat_alpha
        # d["xpander"][int(i*num_of_switches)] = xpander_alpha
        # d["jellyfish"][int(i*num_of_switches)] = jellyfish_alpha
        # d1["strat"][int(i*num_of_switches)] = strat_alpha / th_up
        # d1["xpander"][int(i*num_of_switches)] = xpander_alpha / th_up
        # d1["jellyfish"][int(i*num_of_switches)] = jellyfish_alpha / th_up
        d2["strat"][int(i*num_of_switches)] = (strat_alpha / th_up)*100
        # d2["xpander"][int(i*num_of_switches)] = (xpander_alpha / th_up)*100
        # d2["jellyfish"][int(i*num_of_switches)] = (jellyfish_alpha / th_up)*100
        # f = open(path1+"new.txt", "a")
        # f1 = open(path1 + "cmd.txt", "a")
        # f1.write("it should be written now")
        # f.write(str(i))
        # f.write(str(" "))
        # f.write(str(int(i*num_of_switches)))
        # f.write(str(" "))
        # f.write(str(strat_alpha))
        # f.write(str(" "))
        # f.write(str(xpander_alpha))
        # f.write(str(" "))
        # f.write(str(jellyfish_alpha))
        # f.write(str(" "))
        # f.write(str(th_up))
        # f.write(str(" "))
        # f.write(str(avsp(strat_paths)))
        # f.write(str(" "))
        # f.write(str(avsp(xpander_paths)))
        # f.write(str(" "))
        # f.write(str(avsp(jellyfish_paths)))
        # f.write("\n")
        # f.close()
# plot_path_lengths(d, "C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/max_min_flow_num_s", "Number Of Servers", "Normalized Throughput")
# plot_path_lengths(d1, "C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/max_min_flow_num_s_norm", "Number Of Servers", "Normalized Throughput")
# plot_path_lengths(d2, "C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/max_min_flow_num_s_norm_per", "Number Of Servers", "Normalized Throughput (%)")


# Maximum flow problem
# d1 = {"strat":{}, "xpander":{}, "jellyfish":{}}
# for i in range(1,7):
#     for p in params:
#         num_of_switches, num_servers, switch_k, num_servers_per_rack = p
#         switch_d = switch_k - num_servers_per_rack
#         strat_file = "StratAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
#         strat_graph = read_graph_from_file(path + strat_file)
#         xpander_file = "XpanderAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
#         xpander_graph = read_graph_from_file(path + xpander_file)
#         jellyfish_file = "JellyfishAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
#         jellyfish_graph = read_graph_from_file(path + jellyfish_file)
#         F = sorted(all_to_all(strat_graph, 1.0, i)).copy()
#         strat_alpha = maximizing_flow(strat_graph, F)
#         xpander_alpha = maximizing_flow(xpander_graph, F)
#         jellyfish_alpha = maximizing_flow(jellyfish_graph, F)
#         d["strat"][float(i)] = strat_alpha
#         d["xpander"][float(i)] = xpander_alpha
#         d["jellyfish"][float(i)] = jellyfish_alpha
#         f = open(path1+"result_max_flow_num_s.txt", "a")
#         f.write(str(i))
#         f.write(str(" "))
#         f.write(str(strat_alpha))
#         f.write(str(" "))
#         f.write(str(xpander_alpha))
#         f.write(str(" "))
#         f.write(str(jellyfish_alpha))
#         f.write("\n")
# plot_path_lengths(d1, "C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/max_flow_num_s", "Number Of Servers", "Normalized Throughput")

# Maximum flow problem
# d2 = {"strat":{}, "xpander":{}, "jellyfish":{}}
# load = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# for i in load:
#     strat = []
#     xpander = []
#     jellyfish = []
#     for j in range(10):
#         for p in params:
#             num_of_switches, num_servers, switch_k, num_servers_per_rack = p
#             switch_d = switch_k - num_servers_per_rack
#             strat_file = "StratAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
#             strat_graph = read_graph_from_file(path + strat_file)
#             xpander_file = "XpanderAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
#             xpander_graph = read_graph_from_file(path + xpander_file)
#             jellyfish_file = "JellyfishAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
#             jellyfish_graph = read_graph_from_file(path + jellyfish_file)
#             F1 = sorted(all_to_all(strat_graph, 1.0)).copy()
#             random.shuffle(F1)
#             number_of_flows_used = int(len(F1)*i)
#             F = random.sample(F1, number_of_flows_used)
#             strat_alpha = maximizing_flow(strat_graph, F)
#             strat.append(strat_alpha)
#             xpander_alpha = maximizing_flow(xpander_graph, F)
#             xpander.append(xpander_alpha)
#             jellyfish_alpha = maximizing_flow(jellyfish_graph, F)
#             jellyfish.append(jellyfish_alpha)
#     d["strat"][i] = np.mean(strat)
#     d["xpander"][i] = np.mean(xpander)
#     d["jellyfish"][i] = np.mean(jellyfish)
#     f = open(path1+"result_max_flow_load.txt", "a")
#     f.write(str(i))
#     f.write(str(" "))
#     f.write(str(np.mean(strat)))
#     f.write(str(" "))
#     f.write(str(np.mean(xpander)))
#     f.write(str(" "))
#     f.write(str(np.mean(jellyfish)))
#     f.write("\n")
# plot_path_lengths(d2, "C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/max_flow_load", "Network Load (%)", "Normalized Throughput")
