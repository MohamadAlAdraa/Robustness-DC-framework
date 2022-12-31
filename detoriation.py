import itertools
import math
import random
from collections import defaultdict
import networkx as nx
from utilities import read_graph_from_file, find_average_shortest_path, find_diameter, find_laplacian_spectrum, find_adjacency_spectrum
topos = ['xpander', 'strat', 'jellyfish']
f_p = 'C:/Users/umroot/PycharmProjects/datacenter/failing_figures/node/node_'
strat_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d15/'
jellyfish_path = 'C:/Users/umroot/PycharmProjects/datacenter/new_jellyfish/d15/'
xpander_path = 'C:/Users/umroot/PycharmProjects/datacenter/new_xpander/d15/'
path = {'strat': strat_path, 'xpander': xpander_path, 'jellyfish': jellyfish_path}
prefix_file_name = {'strat': 'StratAL', 'xpander': 'XpanderAL', 'jellyfish': 'JellyfishAL'}


def random_failure(number_of_nodes_to_be_removed, nodes):
    return random.sample(nodes, k=number_of_nodes_to_be_removed)


def sort_dict_by_value_lowest_to_highest(dict):
    return {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}


def target_failure(number_of_nodes_to_be_removed, G):
    betw_dic = nx.betweenness_centrality(G)
    d = sort_dict_by_value_lowest_to_highest(betw_dic)
    nodes_to_be_removed = list(d.keys())[:number_of_nodes_to_be_removed]
    return nodes_to_be_removed


def det_helper(d, d1, metric):
    l = defaultdict(defaultdict)
    for i,j in d.items():
        for x,y in j.items():
            in_ = d1[i][metric]
            l[i][x] = abs(((1/in_)*(float(y)-in_))*100)
    return l


def print_d(d):
    for i, j in d.items():
        print(i)
        for x,y in j.items():
            print(x, float("{0:.5f}". format(float(y))))

def detoriation(params, failure_type, failure_rate, number_of_iterations):
    for p in params:
        shortest_path_dict = defaultdict(defaultdict)
        # diameter_dict = defaultdict(defaultdict)
        laplacian_spectrum_dict = defaultdict(defaultdict)
        adjacency_spectrum_dict = defaultdict(defaultdict)
        num_of_switches, num_servers, switch_k, num_servers_per_rack = p
        initial_values_t = defaultdict(defaultdict)
        for t in range(len(topos)):
            # print(p, topos[t])
            f = prefix_file_name[topos[t]] + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
            f_path = path[topos[t]]
            in_G = read_graph_from_file(f_path + f)
            initial_values = {"avsp": find_average_shortest_path(in_G), "diameter": find_diameter(in_G),
                              "alg_con": find_laplacian_spectrum(in_G), "spectral_gap": (find_adjacency_spectrum(in_G)[0] - find_adjacency_spectrum(in_G)[1])}
            initial_values_t[topos[t]] = initial_values
            for r in failure_rate:
                number_of_nodes_to_be_removed = math.floor(r * num_of_switches)
                av_sh = 0
                av_dm = 0
                av_lp = 0
                av_ad = 0
                for i in range(number_of_iterations):
                    # print(prefix_file_name[topos[t]], r, i)
                    b = True
                    G = nx.Graph()
                    while b:
                        G1 = read_graph_from_file(f_path + f)
                        nodes = [n for n in G1.nodes()]
                        nodes_to_be_removed = random_failure(number_of_nodes_to_be_removed, nodes) if failure_type == 'r' else target_failure(number_of_nodes_to_be_removed, G1)
                        for rm in nodes_to_be_removed:
                            G1.remove_node(rm)
                        if nx.is_connected(G1):
                            b = False
                            G = G1.copy()

                    av_sh += find_average_shortest_path(G)
                    av_dm += find_diameter(G)
                    ls2 = find_laplacian_spectrum(G)
                    av_lp += ls2.real
                    ad1, ad2 = find_adjacency_spectrum(G)
                    ad = ad1 - ad2
                    av_ad += ad.real
                shortest_path_dict[topos[t]][float("{:.1f}".format(r*100))] = "{:.5f}".format(av_sh / number_of_iterations)
                # diameter_dict[topos[t]][float("{:.1f}".format(r * 100))] = "{:.5f}".format(av_dm / number_of_iterations)
                laplacian_spectrum_dict[topos[t]][float("{:.1f}".format(r * 100))] = "{:.5f}".format(av_lp / number_of_iterations)
                adjacency_spectrum_dict[topos[t]][float("{:.1f}".format(r * 100))] = "{:.5f}".format(av_ad / number_of_iterations)
        # print(shortest_path_dict)
        # print(diameter_dict)
        # print(laplacian_spectrum_dict)
        # print(adjacency_spectrum_dict)
        # print(initial_values_t)
        n = 1
        # print(initial_values_t)
        print("AVSP:")
        print_d(shortest_path_dict)
        print_d(det_helper(shortest_path_dict, initial_values_t, "avsp"))
        print("###########################################################")
        print_d(laplacian_spectrum_dict)
        print_d(det_helper(laplacian_spectrum_dict, initial_values_t, "alg_con"))
        print("###########################################################")
        print_d(adjacency_spectrum_dict)
        print_d(det_helper(adjacency_spectrum_dict, initial_values_t, "spectral_gap"))




p = [(256,2048,23,8)]
# p1 = [(512,4096,32,8)]
fr = [0.02,0.04,0.06,0.08,0.1,0.12]
target = 't'
random_ = 'r'
print("###########################################################")
print("256, r, 1000")
detoriation(p, random_, fr, 1000)
print("###########################################################")
print("256, t, 1")
detoriation(p, target, fr, 1)
# print('\n')
# print("###########################################################")
# print('\n')
# print("###########################################################")
# print("512, r, 100")
# detoriation(p1, random_, fr, 500)
# print("###########################################################")
# print("512, t, 1")
# detoriation(p, target, fr, 1)
