import networkx as nx
import random
from utilities import read_graph_from_file, find_average_shortest_path, plot_path_lengths, find_diameter, find_laplacian_spectrum, find_adjacency_spectrum, find_betweenness_centrality, find_edge_betweenness_centrality, find_std_av_shortest_path_length
import math
from collections import defaultdict
import json

params = [(512,4096,32,8)]
topos = ['xpander', 'strat', 'jellyfish']
f_p = 'C:/Users/umroot/PycharmProjects/datacenter/failing_figures/node/node_'
strat_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d24/'
jellyfish_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_jellyfish/d24/'
xpander_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_xpander/d24/'
path = {'strat': strat_path, 'xpander': xpander_path, 'jellyfish': jellyfish_path}
adl = {'strat': 'StratAL', 'xpander': 'XpanderAL', 'jellyfish': 'JellyfishAL'}
failure_rate = [0.01,0.02,0.03,0.04,0.05,0.06]
number_of_iterations = 500

for p in params:
    shortest_path_dict = defaultdict(defaultdict)
    shortest_path_dict1 = defaultdict(defaultdict)
    diameter_dict = defaultdict(defaultdict)
    laplacian_spectrum_dict = defaultdict(defaultdict)
    laplacian_spectrum_dict1 = defaultdict(defaultdict)
    adjacency_spectrum_dict = defaultdict(defaultdict)
    adjacency_spectrum_dict1 = defaultdict(defaultdict)
    betweenness_centrality_dict = defaultdict(defaultdict)
    edge_betweenness_centrality_dict = defaultdict(defaultdict)
    disc_prob_dict = defaultdict(defaultdict)
    num_of_switches, num_servers, switch_k, num_servers_per_rack = p
    switch_d = switch_k - num_servers_per_rack
    switch_degree = switch_d
    for t in range(len(topos)):
        f = adl[topos[t]] + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
        t_path = path[topos[t]]
        for r in failure_rate:
            number_of_nodes_to_be_removed = math.floor(r*num_of_switches)
            av_sh = 0
            shortest_path_av_std_list = []
            av_dm = 0
            laplacian_spectrum_list = []
            laplacian_spectrum = 0
            adjacency_spectrum_list = []
            betweenness_centrality_list = []
            edge_betweenness_centrality_list = []
            adjacency_spectrum = 0
            d_p_c = 0
            for i in range(number_of_iterations):
                print(adl[topos[t]], r, i)
                b = True
                G = nx.Graph()
                while b:
                    d_p_c += 1
                    G1 = read_graph_from_file(t_path + f)
                    nodes = [n for n in G1.nodes()]
                    nodes_to_be_removed = random.sample(nodes, k=number_of_nodes_to_be_removed)
                    for rm in nodes_to_be_removed:
                        G1.remove_node(rm)
                    if(nx.is_connected(G1)):
                        b=False
                        G = G1.copy()
                av_sh += find_average_shortest_path(G)
                shortest_path_av_std_list.append([find_average_shortest_path(G), find_std_av_shortest_path_length(G)])
                av_dm += find_diameter(G)
                ls2 = find_laplacian_spectrum(G)
                laplacian_spectrum_list.append("{:.5f}".format(ls2.real))
                laplacian_spectrum += ls2.real
                al1, al2 = find_adjacency_spectrum(G)
                ad = al1 - al2
                adjacency_spectrum_list.append("{:.5f}".format(ad.real))
                adjacency_spectrum += ad.real
                max_bc, min_bc, av_bc, std_bc, b = find_betweenness_centrality(G)
                max_ebc, min_ebc, av_ebc, std_ebc, eb = find_edge_betweenness_centrality(G)
                betweenness_centrality_list.append([max_bc, min_bc, av_bc, std_bc])
                edge_betweenness_centrality_list.append([max_ebc, min_ebc, av_ebc, std_ebc])
            disc_prob_dict[topos[t]][float("{:.1f}".format(r*100))] = d_p_c
            shortest_path_dict[topos[t]][float("{:.1f}".format(r*100))] = "{:.5f}".format(av_sh/number_of_iterations)
            diameter_dict[topos[t]][float("{:.1f}".format(r * 100))] = "{:.5f}".format(av_dm / number_of_iterations)
            laplacian_spectrum_dict[topos[t]][float("{:.1f}".format(r * 100))] = laplacian_spectrum_list
            laplacian_spectrum_dict1[topos[t]][float("{:.1f}".format(r * 100))] = "{:.5f}".format(laplacian_spectrum/number_of_iterations)
            adjacency_spectrum_dict[topos[t]][float("{:.1f}".format(r * 100))] = adjacency_spectrum_list
            adjacency_spectrum_dict1[topos[t]][float("{:.1f}".format(r * 100))] = "{:.5f}".format(adjacency_spectrum/number_of_iterations)
            betweenness_centrality_dict[topos[t]][float("{:.1f}".format(r * 100))] = betweenness_centrality_list
            edge_betweenness_centrality_dict[topos[t]][float("{:.1f}".format(r * 100))] = edge_betweenness_centrality_list
            shortest_path_dict1[topos[t]][float("{:.1f}".format(r * 100))] = shortest_path_av_std_list

    with open(f_p + "file_av_shortest_path_failing" + "_" + str(num_of_switches) + '.json', 'w') as f:
        json.dump(shortest_path_dict, f)
    with open(f_p + "file_av_shortest_path_std_failing" + "_" + str(num_of_switches) + '.json', 'w') as f:
        json.dump(shortest_path_dict1, f)
    with open(f_p + "file_diameter_failing" + "_" + str(num_of_switches) + '.json', 'w') as f:
        json.dump(diameter_dict, f)
    with open(f_p + "file_laplacian_failing" + "_" + str(num_of_switches) + '.json', 'w') as f:
        json.dump(laplacian_spectrum_dict, f)
    with open(f_p + "file_adjacency_failing" + "_" + str(num_of_switches) + '.json', 'w') as f:
        json.dump(adjacency_spectrum_dict, f)
    with open(f_p + "file_laplacian1_failing" + "_" + str(num_of_switches) + '.json', 'w') as f:
        json.dump(laplacian_spectrum_dict1, f)
    with open(f_p + "file_adjacency1_failing" + "_" + str(num_of_switches) + '.json', 'w') as f:
        json.dump(adjacency_spectrum_dict1, f)
    with open(f_p + "file_betw_failing" + "_" + str(num_of_switches) + '.json', 'w') as f:
        json.dump(betweenness_centrality_dict, f)
    with open(f_p + "file_edge_betw_failing" + "_" + str(num_of_switches) + '.json', 'w') as f:
        json.dump(edge_betweenness_centrality_dict, f)
    with open(f_p + "file_disp_failing" + "_" + str(num_of_switches) + '.json', 'w') as f:
        json.dump(disc_prob_dict, f)
    plot_path_lengths(shortest_path_dict, f_p +"avsp_" + str(num_of_switches), 'Percentage of failing nodes', 'Av. shortest path')
    plot_path_lengths(diameter_dict, f_p + "diameter_" + str(num_of_switches), 'Percentage of failing nodes', 'Diameter')
    plot_path_lengths(laplacian_spectrum_dict1, f_p + "lapl_" + str(num_of_switches), 'Percentage of failing nodes', 'Laplacian spectrum')
    plot_path_lengths(adjacency_spectrum_dict1, f_p + "adj_" + str(num_of_switches), 'Percentage of failing nodes', 'Spectral gap')




