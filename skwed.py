import networkx as nx

import pulp as pl
from utilities import read_graph_from_file


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


def find_shortest_path(G, t):
    dct = {}
    for i in t:
        s, d, id = i
        path_length_list = nx.shortest_path(G, source=s, target=d)
        path_length = len(path_length_list) - 1
        if path_length not in list(dct.keys()):
            dct[path_length] = [(s, d, id)]
        else:
            dct[path_length].append((s, d, id))
    return dct


def skwed_matrix(G, n):
    x = all_to_all(G, n)
    t = find_shortest_path(G, x)
    sk_traffic = []
    l = dict()
    for i, j in t.items():
        l[i] = len(j)
        for k in j:
            s, d, id = k
            sk_traffic.append((s, d, 1/i, id))
    return sk_traffic, l


def skwed_matrix1(G, l, n):
    x = all_to_all(G, n)
    t = find_shortest_path(G, x)
    sk_traffic_temp = []
    for i, j in t.items():
        for k in j:
            sk_traffic_temp.append(k)
    sk_traffic = []
    ind = 0
    for i, j in l.items():
        for k in range(ind, ind+j):
            s, d, id = sk_traffic_temp[k]
            sk_traffic.append((s, d, 1/i, id))
        ind = j

    return sk_traffic


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
            flow_edges[(e,f)] = pl.LpVariable(variable_name, lowBound=0, upBound=f[2], cat=pl.LpContinuous)

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
                mcf_model += ((pl.lpSum(Cs1)) - (pl.lpSum(Cs))) == alpha * f[2]
            elif v == f[1]:
                mcf_model += ((pl.lpSum(Cs1)) - (pl.lpSum(Cs))) == -1.0 * alpha * f[2]
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


G = read_graph_from_file("C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d15/StratAL512_8.txt")
G1 = read_graph_from_file("C:/Users/umroot/PycharmProjects/datacenter/new_xpander/d15/XpanderAL512_8.txt")
G2 = read_graph_from_file("C:/Users/umroot/PycharmProjects/datacenter/new_jellyfish/d15/JellyfishAL512_8.txt")


s, l = skwed_matrix(G, 1)
s1 = skwed_matrix1(G1, l, 1)
s2 = skwed_matrix1(G2, l, 1)

optimal_routing(G, s)
optimal_routing(G1, s1)
optimal_routing(G2, s2)


