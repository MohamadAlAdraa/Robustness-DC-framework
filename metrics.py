import matplotlib
matplotlib.use('Agg')
from collections import defaultdict
import json
from utilities import find_average_shortest_path, draw_graph, read_graph_from_file, lb_av_shortest_path, plot_path_lengths, find_betweenness_centrality, find_edge_betweenness_centrality, find_diameter, find_adjacency_spectrum, find_laplacian_spectrum, jain_fairness_index
import math


#############
strat_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d24/'
jellyfish_path = 'C:/Users/umroot/PycharmProjects/datacenter/new_jellyfish/d15/'
xpander_path = 'C:/Users/umroot/PycharmProjects/datacenter/new_xpander/d15/'
#############

drawn_graphs_path = 'C:/Users/umroot/PycharmProjects/datacenter/drawn_graphs/'
shortest_path_path = 'C:/Users/umroot/PycharmProjects/datacenter/metrics/shortest_path/'
diameter_path = 'C:/Users/umroot/PycharmProjects/datacenter/metrics/diameter/'
betweenness_centrality_path = 'C:/Users/umroot/PycharmProjects/datacenter/metrics/betweenness_centrality/'
edge_betweenness_centrality_path = 'C:/Users/umroot/PycharmProjects/datacenter/metrics/edge_betweenness_centrality/'
adjacency_spectrum_path = 'C:/Users/umroot/PycharmProjects/datacenter/metrics/adjacency_spectrum/'
laplacian_spectrum_path = 'C:/Users/umroot/PycharmProjects/datacenter/metrics/laplacian_spectrum/'
eign_values_path = 'C:/Users/umroot/PycharmProjects/datacenter/eign_values/'

params = [(32,256,23,8),(64,512,23,8),(128,1024,23,8),(256,2048,23,8),(512,4096,32,8)]
#(32,256,23,8)
#######################################################################
# Shortest path dicts
shortest_path_dict = defaultdict(defaultdict) # avsp for each topo and the lb
shortest_path_diff_dict = defaultdict(defaultdict) # avsp for each topo sub from the lb ((avspf - avsplb)/avsplb)*100 to see how much far the avsp from the lb in %
shortest_path_dict_gain = defaultdict(defaultdict) # avsp for each topo sub from the strat ((avspf - avspst)/avspst)*100 to see how much far the avsp from the strat in %
#######################################################################
# Diameter dicts
diameter_dict = defaultdict(defaultdict) # avsp for each topo and the lb
#######################################################################
betweenness_centrality_dict = defaultdict(defaultdict) # list of the betweenness of each node.
betweenness_centrality_dict_jfi = defaultdict(defaultdict) # jain index for the betw of the nodes as it is closed to one that means betw values are more closed to each other
betweenness_centrality_dict_av = defaultdict(defaultdict) # av. betw
betweenness_centrality_dict_max = defaultdict(defaultdict) # max betw
betweenness_centrality_dict_min_max = defaultdict(defaultdict) # max-min betw
#######################################################################
edge_betweenness_centrality_dict = defaultdict(defaultdict) # list of the edge betw of each edge
edge_betweenness_centrality_dict_jfi = defaultdict(defaultdict) # jain index for the betw of the edges as it is closed to one that means betw values are more closed to each other
edge_betweenness_centrality_dict_av = defaultdict(defaultdict) # av. edge betw
edge_betweenness_centrality_dict_max = defaultdict(defaultdict) # max edge betw
edge_betweenness_centrality_dict_min_max = defaultdict(defaultdict) # max-min edge betw
#######################################################################
adjacency_spectrum_dict_diff = defaultdict(defaultdict) # spectral gap which (largest - second_largest) eigen values of the adj matrix
adjacency_spectrum_dict_diff_gain = defaultdict(defaultdict) # sp for each topo sub from the strat ((spf - sps)/sps)*100 to see how much gain strat has over other topos in %
adjacency_spectrum_dict_second_largest = defaultdict(defaultdict) # second largest eigen value in abs
#######################################################################
laplacian_spectrum_dict = defaultdict(defaultdict) # second smallest eigen value of the laplacian matrix (algb connectivity)
laplacian_spectrum_dict_gain = defaultdict(defaultdict) # ss for each topo sub from the strat ((ssf - sss)/sss)*100 to see how much gain strat has over other topos in %
#######################################################################
eign_values_dict = defaultdict(defaultdict) # has the largest, second_largest of the adj and second smallest of the laplacien for each graph
#######################################################################

for p in params:
    num_of_switches, num_servers, switch_k, num_servers_per_rack = p
    switch_d = switch_k - num_servers_per_rack
    switch_degree = switch_d
    #########################
    # STRAT TOPOLOGY
    strat_file = "StratAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
    strat_graph = read_graph_from_file(strat_path+strat_file)
    draw_graph(strat_graph, drawn_graphs_path +'strat_graph'+str(num_of_switches) + str(switch_degree)+ "_24")
    shortest_path_dict["STRAT"][num_of_switches] = "{:.5f}".format(find_average_shortest_path(strat_graph).real)
    shortest_path_diff_dict["STRAT"][num_of_switches] = "{:.5f}".format(((find_average_shortest_path(strat_graph).real - lb_av_shortest_path(num_of_switches, switch_degree).real)/lb_av_shortest_path(num_of_switches, switch_degree).real)*100)
    diameter_dict["STRAT"][num_of_switches] = "{:.5f}".format(find_diameter(strat_graph).real)
    max_bc, min_bc, av_bc, std_, b = find_betweenness_centrality(strat_graph)
    max_ebc, min_ebc, av_ebc, std_, eb = find_edge_betweenness_centrality(strat_graph)
    betweenness_centrality_dict["STRAT"][num_of_switches] = list(b.values())
    betweenness_centrality_dict_av["STRAT"][num_of_switches] = "{:.5f}".format(av_bc.real)
    edge_betweenness_centrality_dict_av["STRAT"][num_of_switches] = "{:.5f}".format(av_ebc.real)
    edge_betweenness_centrality_dict["STRAT"][num_of_switches] = list(eb.values())
    betweenness_centrality_dict_max["STRAT"][num_of_switches] = "{:.5f}".format(max_bc.real)
    edge_betweenness_centrality_dict_max["STRAT"][num_of_switches] = "{:.5f}".format(max_ebc.real)
    betweenness_centrality_dict_min_max["STRAT"][num_of_switches] = "{:.5f}".format(max_bc.real - min_bc.real)
    edge_betweenness_centrality_dict_min_max["STRAT"][num_of_switches] = "{:.5f}".format(max_ebc.real - min_ebc.real)
    betweenness_centrality_dict_jfi["STRAT"][num_of_switches] = "{:.5f}".format(jain_fairness_index(b))
    edge_betweenness_centrality_dict_jfi["STRAT"][num_of_switches] = "{:.5f}".format(jain_fairness_index(eb))
    al1, al2 = find_adjacency_spectrum(strat_graph)
    ad = al1-al2
    adjacency_spectrum_dict_second_largest["STRAT"][num_of_switches] = "{:.5f}".format(al2.real)
    adjacency_spectrum_dict_diff["STRAT"][num_of_switches] = "{:.5f}".format(ad.real)
    ls2 = find_laplacian_spectrum(strat_graph)
    laplacian_spectrum_dict["STRAT"][num_of_switches] = "{:.5f}".format(ls2.real)
    eign_values_dict["STRAT"][num_of_switches] = {"adjacency": {"l1" : al1.real, "l2" : al2.real}, "laplacian": {"s2" : ls2.real}}
    b_to_be_used = max_bc
    eb_to_be_used = max_ebc
    ad_to_be_used = ad
    ls2_to_be_used = ls2
    #########################
    # XPANDER TOPOLOGY depends on adjacency
    xpander_file = "XpanderAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
    xpander_graph = read_graph_from_file(xpander_path+xpander_file)
    #draw_graph(xpander_graph, drawn_graphs_path + 'xpander_graph'+str(num_of_switches))
    shortest_path_dict["Xpander"][num_of_switches] = "{:.5f}".format(find_average_shortest_path(xpander_graph).real)
    shortest_path_diff_dict["Xpander"][num_of_switches] = "{:.5f}".format(((find_average_shortest_path(xpander_graph).real - lb_av_shortest_path(num_of_switches, switch_degree).real)/lb_av_shortest_path(num_of_switches, switch_degree).real)*100)
    shortest_path_dict_gain["Xpander"][num_of_switches] = "{:.5f}".format(((find_average_shortest_path(xpander_graph).real - find_average_shortest_path(strat_graph).real)/find_average_shortest_path(strat_graph).real)*100)
    diameter_dict["Xpander"][num_of_switches] = "{:.5f}".format(find_diameter(xpander_graph).real)
    max_bc, min_bc, av_bc, std_, b = find_betweenness_centrality(xpander_graph)
    max_ebc, min_ebc, av_ebc, std_, eb = find_edge_betweenness_centrality(xpander_graph)
    betweenness_centrality_dict_av["Xpander"][num_of_switches] = "{:.5f}".format(av_bc.real)
    betweenness_centrality_dict["Xpander"][num_of_switches] = list(b.values())
    edge_betweenness_centrality_dict_av["Xpander"][num_of_switches] = "{:.5f}".format(av_ebc.real)
    edge_betweenness_centrality_dict["Xpander"][num_of_switches] = list(eb.values())
    betweenness_centrality_dict_max["Xpander"][num_of_switches] = "{:.5f}".format(max_bc.real)
    edge_betweenness_centrality_dict_max["Xpander"][num_of_switches] = "{:.5f}".format(max_ebc.real)
    betweenness_centrality_dict_min_max["Xpander"][num_of_switches] = "{:.5f}".format(max_bc.real - min_bc.real)
    edge_betweenness_centrality_dict_min_max["Xpander"][num_of_switches] = "{:.5f}".format(max_ebc.real - min_ebc.real)
    betweenness_centrality_dict_jfi["Xpander"][num_of_switches] = "{:.5f}".format(jain_fairness_index(b))
    edge_betweenness_centrality_dict_jfi["Xpander"][num_of_switches] = "{:.5f}".format(jain_fairness_index(eb))
    al1, al2 = find_adjacency_spectrum(xpander_graph)
    ad = al1-al2
    adjacency_spectrum_dict_second_largest["Xpander"][num_of_switches] = "{:.5f}".format(al2.real)
    adjacency_spectrum_dict_diff["Xpander"][num_of_switches] = "{:.5f}".format(ad.real)
    adjacency_spectrum_dict_diff_gain["Xpander"][num_of_switches] = "{:.5f}".format(((ad_to_be_used - ad.real)/ad.real)*100)
    ls2 = find_laplacian_spectrum(xpander_graph)
    laplacian_spectrum_dict["Xpander"][num_of_switches] = "{:.5f}".format(ls2.real)
    laplacian_spectrum_dict_gain["Xpander"][num_of_switches] = "{:.5f}".format(((ls2_to_be_used - ls2)/ls2)*100)
    eign_values_dict["Xpander"][num_of_switches] = {"adjacency": {"l1" : al1.real, "l2" : al2.real}, "laplacian": {"s2" : ls2.real}}
    #########################
    # JELLYFISH TOPOLOGY
    jellyfish_file = "JellyfishAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
    jellyfish_graph = read_graph_from_file(jellyfish_path+jellyfish_file)
    #draw_graph(jellyfish_graph, drawn_graphs_path + 'jellyfish_graph'+str(num_of_switches))
    shortest_path_dict["Jellyfish"][num_of_switches] = "{:.5f}".format(find_average_shortest_path(jellyfish_graph).real)
    shortest_path_diff_dict["Jellyfish"][num_of_switches] = "{:.5f}".format(((find_average_shortest_path(jellyfish_graph).real - lb_av_shortest_path(num_of_switches, switch_degree).real)/lb_av_shortest_path(num_of_switches, switch_degree).real)*100)
    shortest_path_dict_gain["Jellyfish"][num_of_switches] = "{:.5f}".format(((find_average_shortest_path(jellyfish_graph).real - find_average_shortest_path(strat_graph).real)/find_average_shortest_path(strat_graph).real)* 100)
    diameter_dict["Jellyfish"][num_of_switches] = "{:.5f}".format(find_diameter(jellyfish_graph).real)
    max_bc, min_bc, av_bc, std_, b = find_betweenness_centrality(jellyfish_graph)
    max_ebc, min_ebc, av_ebc, std_, eb = find_edge_betweenness_centrality(jellyfish_graph)
    betweenness_centrality_dict_av["Jellyfish"][num_of_switches] = "{:.5f}".format(av_bc.real)
    betweenness_centrality_dict["Jellyfish"][num_of_switches] = list(b.values())
    edge_betweenness_centrality_dict_av["Jellyfish"][num_of_switches] = "{:.5f}".format(av_ebc.real)
    edge_betweenness_centrality_dict["Jellyfish"][num_of_switches] = list(eb.values())
    betweenness_centrality_dict_max["Jellyfish"][num_of_switches] = "{:.5f}".format(max_bc.real)
    edge_betweenness_centrality_dict_max["Jellyfish"][num_of_switches] = "{:.5f}".format(max_ebc.real)
    betweenness_centrality_dict_min_max["Jellyfish"][num_of_switches] = "{:.5f}".format(max_bc.real - min_bc.real)
    edge_betweenness_centrality_dict_min_max["Jellyfish"][num_of_switches] = "{:.5f}".format(max_ebc.real - min_ebc.real)
    betweenness_centrality_dict_jfi["Jellyfish"][num_of_switches] = "{:.5f}".format(jain_fairness_index(b))
    edge_betweenness_centrality_dict_jfi["Jellyfish"][num_of_switches] = "{:.5f}".format(jain_fairness_index(eb))
    al1, al2 = find_adjacency_spectrum(jellyfish_graph)
    ad = al1-al2
    adjacency_spectrum_dict_second_largest["Jellyfish"][num_of_switches] = "{:.5f}".format(al2.real)
    adjacency_spectrum_dict_diff["Jellyfish"][num_of_switches] = "{:.5f}".format(ad.real)
    adjacency_spectrum_dict_diff_gain["Jellyfish"][num_of_switches] = "{:.5f}".format(((ad_to_be_used - ad.real)/ad.real)*100)
    ls2 = find_laplacian_spectrum(jellyfish_graph)
    laplacian_spectrum_dict["Jellyfish"][num_of_switches] = "{:.5f}".format(ls2.real)
    laplacian_spectrum_dict_gain["Jellyfish"][num_of_switches] = "{:.5f}".format(((ls2_to_be_used - ls2)/ls2)*100)
    eign_values_dict["Jellyfish"][num_of_switches] = {"adjacency": {"l1" : al1.real, "l2" : al2.real}, "laplacian": {"s2" : ls2.real}}
    #########################
    shortest_path_dict["Opt"][num_of_switches] = "{:.5f}".format(lb_av_shortest_path(num_of_switches, switch_degree).real)
    diameter_dict["Opt"][num_of_switches] = "{:.5f}".format(math.ceil(math.log(num_of_switches, switch_degree)).real)

# with open(shortest_path_path + "file_av_shortest_path" + '.json', 'w') as f:
#     json.dump(shortest_path_dict, f)

# with open(shortest_path_path + "file_av_shortest_path_sub_from_lb" + '.json', 'w') as f:
#     json.dump(shortest_path_diff_dict, f)

with open(shortest_path_path + "file_av_shortest_path_sub_from_st" + '.json', 'w') as f:
    json.dump(shortest_path_dict_gain, f)

with open(diameter_path + "file_diameter" + '.json', 'w') as f:
    json.dump(diameter_dict, f)

with open(betweenness_centrality_path + "file_betweenness_centrality_av" + '.json', 'w') as f:
    json.dump(betweenness_centrality_dict_av, f)

with open(betweenness_centrality_path + "file_betweenness_centrality_jfi" + '.json', 'w') as f:
    json.dump(betweenness_centrality_dict_jfi, f)

with open(betweenness_centrality_path + "file_betweenness_centrality_max" + '.json', 'w') as f:
    json.dump(betweenness_centrality_dict_max, f)

with open(betweenness_centrality_path + "file_betweenness_centrality" + '.json', 'w') as f:
    json.dump(betweenness_centrality_dict, f)

with open(betweenness_centrality_path + "file_betweenness_centrality_max_min" + '.json', 'w') as f:
    json.dump(betweenness_centrality_dict_min_max, f)

with open(edge_betweenness_centrality_path + "file_edge_betweenness_centrality_av" + '.json', 'w') as f:
    json.dump(edge_betweenness_centrality_dict_av, f)

with open(edge_betweenness_centrality_path + "file_edge_betweenness_centrality_jfi" + '.json', 'w') as f:
    json.dump(edge_betweenness_centrality_dict_jfi, f)

with open(edge_betweenness_centrality_path + "file_edge_betweenness_centrality_max" + '.json', 'w') as f:
    json.dump(edge_betweenness_centrality_dict_max, f)

with open(edge_betweenness_centrality_path + "file_edge_betweenness_centrality" + '.json', 'w') as f:
    json.dump(edge_betweenness_centrality_dict, f)

with open(edge_betweenness_centrality_path + "file_edge_betweenness_centrality_max_min" + '.json', 'w') as f:
    json.dump(edge_betweenness_centrality_dict_min_max, f)

with open(adjacency_spectrum_path + "file_adjacency_spectrum_second_largest" + '.json', 'w') as f:
    json.dump(adjacency_spectrum_dict_second_largest, f)

# with open(adjacency_spectrum_path + "file_adjacency_spectrum_diff" + '.json', 'w') as f:
#     json.dump(adjacency_spectrum_dict_diff, f)

with open(adjacency_spectrum_path + "file_adjacency_spectrum_diff_gain" + '.json', 'w') as f:
    json.dump(adjacency_spectrum_dict_diff_gain, f)

# with open(laplacian_spectrum_path + "file_laplacian_spectrum" + '.json', 'w') as f:
#     json.dump(laplacian_spectrum_dict, f)

with open(laplacian_spectrum_path + "file_laplacian_spectrum_gain" + '.json', 'w') as f:
    json.dump(laplacian_spectrum_dict_gain, f)

with open(eign_values_path + "eign_values" + '.json', 'w') as f:
    json.dump(eign_values_dict, f)

shortest_path_diff_dict = defaultdict(defaultdict)
shortest_path_diff_dict1 = defaultdict(defaultdict)
f = open(shortest_path_path + "file_av_shortest_path_sub_from_lb" + '.json', 'r')
f1 = open(adjacency_spectrum_path + "file_adjacency_spectrum_diff" + '.json', 'r')
f2 = open(laplacian_spectrum_path + "file_laplacian_spectrum" + '.json', 'r')
f3 = open(shortest_path_path + "file_av_shortest_path" + '.json', 'r')
# returns JSON object as
# a dictionary
data = json.load(f)
for i, j in data.items():
    for x, y in j.items():
        shortest_path_diff_dict[i][int(x)] = y
data = json.load(f1)
for i, j in data.items():
    for x, y in j.items():
        adjacency_spectrum_dict_diff[i][int(x)] = y
data = json.load(f2)
for i, j in data.items():
    for x, y in j.items():
        laplacian_spectrum_dict[i][int(x)] = y
data = json.load(f3)
for i, j in data.items():
    for x, y in j.items():
        shortest_path_diff_dict1[i][int(x)] = y

plot_path_lengths(shortest_path_diff_dict1, shortest_path_path+'avsp_graph', 'Number of switches', 'Av. shortest path')
plot_path_lengths(shortest_path_diff_dict, shortest_path_path+'avsp_graph_sub_lb', 'Number of Switches', '\overline{L}_{sp}')
plot_path_lengths(shortest_path_dict_gain, shortest_path_path+'avsp_graph_gain', 'Number of switches', 'Gain (%)')
plot_path_lengths(diameter_dict, diameter_path+'diameter_graph', 'Number of Switches', 'D')
plot_path_lengths(betweenness_centrality_dict_av, betweenness_centrality_path+'betweenness_centrality_graph_av', 'Number of switches', 'Av. Betweenness Centrality')
plot_path_lengths(betweenness_centrality_dict_max, betweenness_centrality_path+'betweenness_centrality_graph_max', 'Number of switches', 'Max. Betweenness Centrality')
plot_path_lengths(betweenness_centrality_dict_min_max, betweenness_centrality_path+'betweenness_centrality_graph_max_min', 'Number of switches', 'Max-Min Betweenness Centrality')
plot_path_lengths(betweenness_centrality_dict_jfi, betweenness_centrality_path+'betweenness_centrality_graph_jfi', 'Number of switches', 'Jain Index of the Betweenness Centrality')
plot_path_lengths(adjacency_spectrum_dict_second_largest, adjacency_spectrum_path+'adjacency_spectrum_second_largest_graph', 'Number of switches', 'Adjacency Spectrum Second Largest')
plot_path_lengths(adjacency_spectrum_dict_diff, adjacency_spectrum_path+'adjacency_spectrum_diff_graph', 'Number of Switches', '\u03BB')
plot_path_lengths(adjacency_spectrum_dict_diff_gain, adjacency_spectrum_path+'adjacency_spectrum_diff_gain_graph', 'Number of switches', 'Gain (%)')
plot_path_lengths(laplacian_spectrum_dict, laplacian_spectrum_path+'laplacian_spectrum_graph', 'Number of Switches', '\u03BB₂')
plot_path_lengths(laplacian_spectrum_dict_gain, laplacian_spectrum_path+'laplacian_spectrum_gain_graph', 'Number of switches', 'Gain of the Laplacian Spectrum')
plot_path_lengths(edge_betweenness_centrality_dict_av, edge_betweenness_centrality_path+'edge_betweenness_centrality_graph_av', 'Number of switches', 'Av. Edge Betweenness Centrality')
plot_path_lengths(edge_betweenness_centrality_dict_max, edge_betweenness_centrality_path+'edge_betweenness_centrality_graph_max', 'Number of switches', 'Max. Edge Betweenness Centrality')
plot_path_lengths(edge_betweenness_centrality_dict_min_max, edge_betweenness_centrality_path+'edge_betweenness_centrality_graph_max_min', 'Number of switches', 'Edge Betweenness Centrality')
plot_path_lengths(edge_betweenness_centrality_dict_jfi, edge_betweenness_centrality_path+'edge_betweenness_centrality_graph_jfi', 'Number of switches', 'Jain Index of the Edge Betweenness Centrality')
