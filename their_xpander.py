# import networkx as nx
# import numpy as np
from utilities import write_graph_to_file, read_graph_from_file, find_diameter, find_adjacency_spectrum, \
    find_laplacian_spectrum, find_average_shortest_path, get_the_best_xpander1, find_std_av_shortest_path_length

# def XP(filename):
#     f = open(filename, 'r')
#     l = f.readlines()
#     ll = []
#     for i in l:
#         ll.append([int(j.strip('\n')) for j in i.split('->')])
#     print(ll)
#     min_node = min(np.array(ll).flatten())
#     max_node = max(np.array(ll).flatten())
#     G = nx.Graph()
#     for i in range(min_node, max_node+1):
#         G.add_node(i)
#     for i in ll:
#         if G.has_edge(i[0], i[1]) or G.has_edge(i[1], i[0]):
#             pass
#         else:
#             G.add_edge(i[0], i[1])
#     print(len(list(G.nodes())))
#     print((len(list(G.edges()))))
#     write_graph_to_file(G, 'xp_their_version')
# XP('xpander_256_12.txt')

# G = read_graph_from_file('XpanderAL120_1.txt')
# G1 = read_graph_from_file('XpanderAL200_1.txt')
G2 = read_graph_from_file('C:/Users/umroot/PycharmProjects/datacenter/ad_list_xpander/d24/XpanderAL2048_8.txt')
# print('120', find_diameter(G))
# print('200', find_diameter(G1))
print('512', find_diameter(G2))
# print('120', find_average_shortest_path(G))
# print('200', find_average_shortest_path(G1))
print('512', find_average_shortest_path(G2))
print('512', find_std_av_shortest_path_length(G2))

# print('120: largest, sec largest adj', find_adjacency_spectrum(G))
# print('200: largest, sec largest adj', find_adjacency_spectrum(G1))
# print('512: largest, sec largest adj', find_adjacency_spectrum(G2))
# print('120: sec smallest lap', find_laplacian_spectrum(G))
# print('200: largest, sec smallest lap', find_laplacian_spectrum(G1))
# print('512: largest, sec smallest lap', find_laplacian_spectrum(G2))

# from utilities import write_graph_to_file, get_the_best_xpander1
#
# params = [(512,512,8,1)]
# xpander_path_a = 'C:/Users/umroot/PycharmProjects/datacenter/new_xpander/'
# for p in params:
#     num_switches, num_servers, switch_k, num_servers_per_rack = p
#     switch_d = switch_k - num_servers_per_rack
#     print("##########################################################")
#     print('Create Xpander with the following params:\n')
#     print('switch ports', switch_k)
#     print('switch degree', switch_d)
#     print('num_switches', num_switches)
#     print('num_servers', num_servers)
#     print('num_servers_per_rack', num_servers_per_rack, '\n')
#     # Xpander graph
#     xpander_graph = get_the_best_xpander1(num_switches, num_servers, num_servers_per_rack, switch_d, 1)
#     write_graph_to_file(xpander_graph, xpander_path_a+"XpanderAL"+str(num_servers)+ "_" + str(num_servers_per_rack) +".txt")
#     print('Xpander ad_list file with', num_servers, 'servers and', num_switches, 'switches has been created')
#     print('Graph validation: ')
#     print('Number of nodes', len(xpander_graph.nodes()))
#     print('Max degree', max(list(xpander_graph.degree())))
#     print('Min degree', min(list(xpander_graph.degree())))
#     print("##########################################################")
