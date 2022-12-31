import random
import math

import networkx as nx
import numpy as np
import pulp as pl

from utilities import read_graph_from_file, plot_path_lengths


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


def all_to_all(G, demand):
    d = []
    i = 0
    for v in G.nodes():
        for v1 in G.nodes():
            if v != v1:
                i += 1
                d.append((v, v1, demand, i))
    return d


def optimal_routing(G, tf):
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
    cplex = pl.get_solver('CPLEX_CMD')
    mcf_model.solve(solver=cplex)

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
    return mcf_model.objective.value()


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


def sort_dict_by_value_lowest_to_highest(dict):
    return {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}


def target_failure(number_of_nodes_to_be_removed, G):
    betw_dic = nx.edge_betweenness_centrality(G)
    d = sort_dict_by_value_lowest_to_highest(betw_dic)
    edges_to_be_removed = list(d.keys())[:number_of_nodes_to_be_removed]
    return edges_to_be_removed


def failure(G, f_r):
    number_of_edges_to_be_removed = math.floor(f_r * len(list(G.edges())))
    G_temp = G.copy()
    edges_to_be_removed = target_failure(number_of_edges_to_be_removed, G_temp)
    edges_to_be_removed_after_f = edges_to_be_removed.copy()
    for rm in edges_to_be_removed:
        G_temp.remove_edge(rm[0], rm[1])
    G_after_f= G_temp.copy()
    return edges_to_be_removed_after_f, G_after_f


path = 'C:/Users/umroot/PycharmProjects/datacenter/topo_for_optimal_routing/topo_used/'
path1 = 'C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/'

params = [(64,512,23,8)]


results = {"strat":{},"xpander":{}, "jellyfish":{}}
# results1 = {"strat":{},"xpander":{}, "jellyfish":{}}
# results2 = {"strat":{},"xpander":{}, "jellyfish":{}}
fail_rate = [0.01,0.03,0.06,0.09,0.12,0.15,0.18,0.21,0.24,0.27,0.3] #,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1
used_alpha = None
topos_name = ["strat", "xpander", "jellyfish"]
topo_graphs = {"strat":{}, "xpander":{}, "jellyfish":{}}
for p in params:
    num_of_switches, num_servers, switch_k, num_servers_per_rack = p
    switch_d = switch_k - num_servers_per_rack
    strat_file = "StratAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
    strat_graph = read_graph_from_file(path + strat_file)
    xpander_file = "XpanderAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
    xpander_graph = read_graph_from_file(path + xpander_file)
    jellyfish_file = "JellyfishAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
    jellyfish_graph = read_graph_from_file(path + jellyfish_file)
    F = sorted(all_to_all(strat_graph, 1.0)).copy()
    # strat_current_topo, strat_alpha, strat_f_values_on_edges, strat_paths, strat_utilization, strat_flows_on_edges = optimal_routing(strat_graph, F)
    strat_alpha = optimal_routing(strat_graph, F)
    F1 = sorted(all_to_all(xpander_graph, 1.0)).copy()
    # xpander_current_topo, xpander_alpha, xpander_f_values_on_edges, xpander_paths, xpander_utilization, xpander_flows_on_edges = optimal_routing(xpander_graph, F1)
    xpander_alpha = optimal_routing(xpander_graph, F1)
    F2 = sorted(all_to_all(jellyfish_graph, 1.0)).copy()
    # jellyfish_current_topo, jellyfish_alpha, jellyfish_f_values_on_edges, jellyfish_paths, jellyfish_utilization, jellyfish_flows_on_edges = optimal_routing(jellyfish_graph, F2)
    jellyfish_alpha = optimal_routing(jellyfish_graph, F2)
    alphas = [strat_alpha, xpander_alpha, jellyfish_alpha]
    used_alpha = min(alphas)
    topo_graphs["strat"]["topo"] = strat_graph
    topo_graphs["xpander"]["topo"] = xpander_graph
    topo_graphs["jellyfish"]["topo"] = jellyfish_graph
    # F = sorted(all_to_all(strat_graph, used_alpha)).copy()
    # strat_current_topo_new, strat_alpha_new, strat_f_values_on_edges_new, strat_paths_new, strat_utilization_new, strat_flows_on_edges_new = optimal_routing(strat_graph, F)
    # F1 = sorted(all_to_all(xpander_graph, used_alpha)).copy()
    # xpander_current_topo_new, xpander_alpha_new, xpander_f_values_on_edges_new, xpander_paths_new, xpander_utilization_new, xpander_flows_on_edges_new = optimal_routing(xpander_graph, F1)
    # F2 = sorted(all_to_all(jellyfish_graph, used_alpha)).copy()
    # jellyfish_current_topo_new, jellyfish_alpha_new, jellyfish_f_values_on_edges_new, jellyfish_paths_new, jellyfish_utilization_new, jellyfish_flows_on_edges_new = optimal_routing(jellyfish_graph, F2)
    # alphas_new = [strat_alpha_new, xpander_alpha_new, jellyfish_alpha_new]
    # topo_graphs["strat"]["topo"] = strat_current_topo_new
    # topo_graphs["strat"]["voe"] = strat_f_values_on_edges_new
    # topo_graphs["strat"]["p"] = strat_paths_new
    # topo_graphs["strat"]["u"] = strat_utilization_new
    # topo_graphs["strat"]["foe"] = strat_flows_on_edges_new
    # topo_graphs["xpander"]["topo"] = xpander_current_topo_new
    # topo_graphs["xpander"]["voe"] = xpander_f_values_on_edges_new
    # topo_graphs["xpander"]["p"] = xpander_paths_new
    # topo_graphs["xpander"]["u"] = xpander_utilization_new
    # topo_graphs["xpander"]["foe"] = xpander_flows_on_edges_new
    # topo_graphs["jellyfish"]["topo"] = jellyfish_current_topo_new
    # topo_graphs["jellyfish"]["voe"] = jellyfish_f_values_on_edges_new
    # topo_graphs["jellyfish"]["p"] = jellyfish_paths_new
    # topo_graphs["jellyfish"]["u"] = jellyfish_utilization_new
    # topo_graphs["jellyfish"]["foe"] = jellyfish_flows_on_edges_new
for topo_name in topos_name:
    G = topo_graphs[topo_name]["topo"].to_directed().copy()
    number_of_rerouted_admitted_flows_initial = maximizing_rerouted_admitted_flows(G, {e: 1.0 for e in sorted(list(G.edges()))}, all_to_all(G, used_alpha))
    results[topo_name][float(0)] = number_of_rerouted_admitted_flows_initial
    for rate in fail_rate:
        temp_result1 = []
        # temp_result2 = []
        for iter in range(1):
            # utilization1 = topo_graphs[topo_name]["u"].copy()
            G = topo_graphs[topo_name]["topo"].copy()
            edges_to_be_removed_after_failure, G_after_failure = failure(G, rate)
            # affected_flows = []
            # for ed_rm in edges_to_be_removed_after_failure:
            #     for f_rm in topo_graphs[topo_name]["foe"][ed_rm]:
            #         if f_rm not in affected_flows:
            #             affected_flows.append(f_rm)
            # for f_rm in affected_flows:
            #     for e_u in topo_graphs[topo_name]["p"][f_rm]:
            #         util_to_be_reduced = topo_graphs[topo_name]["voe"]['f{}_{}_{}_{}_{}_{}'.format(e_u[0], e_u[1], f_rm[0], f_rm[1], f_rm[2], f_rm[3])]
            #         temp_util = utilization1[e_u]
            #         utilization1[e_u] = temp_util - util_to_be_reduced
            # for ed_rm in edges_to_be_removed_after_failure:
            #     del utilization1[ed_rm]
            # C = {k: abs(1-v) for k, v in utilization1.items()}
            # new_traffic_matrix = []
            # for i in range(len(affected_flows)):
            #     new_traffic_matrix.append((affected_flows[i][0], affected_flows[i][1], used_alpha, i))
            G1 = G_after_failure.to_directed().copy()
            # number_of_rerouted_admitted_flows = maximizing_rerouted_admitted_flows(G1, C, new_traffic_matrix)
            number_of_rerouted_admitted_flows = maximizing_rerouted_admitted_flows(G1, {e: 1.0 for e in sorted(list(G1.edges()))}, all_to_all(G1, used_alpha))
            temp_result1.append(int(number_of_rerouted_admitted_flows))
            # temp_result2.append(len(new_traffic_matrix))
        results[topo_name][rate] = np.mean(temp_result1)
        # results1[topo_name][rate] = np.mean(temp_result2)
        # results2[topo_name][rate] = (np.mean(temp_result1)*100)/np.mean(temp_result2)
        f = open(path1 + "result_max_adm_flow_fr.txt", "a")
        f.write(str(topo_name))
        f.write(str(" "))
        f.write(str(rate))
        f.write(str(" "))
        f.write(str(np.mean(temp_result1)))
        # f.write(str(" "))
        # f.write(str(np.mean(temp_result2)))
        # f.write(str(" "))
        # f.write(str((np.mean(temp_result1)*100)/np.mean(temp_result2)))
        f.write("\n")

plot_path_lengths(results, "C:/Users/umroot/PycharmProjects/datacenter/optimal_routing_results/max_adm_flows", "Failure Rate (%)", "Number Of Admitted Flows")