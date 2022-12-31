# import networkx as nx
# import nxmetis
# def bisection_bandwidth(G):
#     a, b = nxmetis.partition(G, 2, recursive=False)
#     a1, b1 = nxmetis.partition(G, 2, recursive=True)
#     return max(a,a1)
#
# strat_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d24/'
# jellyfish_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_jellyfish/d24/'
# xpander_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_xpander/d24/'
# params = [(32,256,32,8),(64,512,32,8),(128,1024,32,8),(256,2048,32,8),(512,4096,32,8),(1024,8192,32,8)]
# d = {'strat':[], 'jellyfish':[], 'xpander':[], 'upper bound': []}
# for p in params:
#     num_of_switches, num_servers, switch_k, num_servers_per_rack = p
#     switch_d = switch_k - num_servers_per_rack
#     switch_degree = switch_d
#     strat_file = "StratAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
#     strat_graph = nx.read_adjlist(strat_path+strat_file)
#     d['strat'].append(bisection_bandwidth(strat_graph))
#     xpander_file = "XpanderAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
#     xpander_graph = nx.read_adjlist(xpander_path+xpander_file)
#     d['xpander'].append(bisection_bandwidth(xpander_graph))
#     jellyfish_file = "JellyfishAL" + str(num_servers) + "_" + str(num_servers_per_rack) + ".txt"
#     jellyfish_graph = nx.read_adjlist(jellyfish_path+jellyfish_file)
#     d['jellyfish'].append(bisection_bandwidth(jellyfish_graph))
#     d['upper bound'].append((num_of_switches*switch_d)/4)


# f = open("AM512P24H3E.txt", "r")
# l = f.readlines()
#
# ll = []
# for i in l:
#     print([int(float(j.strip('\n'))) for j in i.split(',')], end=";")
from strat import Strat
from utilities import write_graph_to_file, read_graph_from_file
# params = [(16,32,8,2), (24,48,8,2), (32,64,8,2), (40,80,8,2), (48,96,8,2), (56,112,8,2), (64,128,8,2), (72,144,8,2), (80,160,8,2), (88,176,8,2), (96,192,8,2), (104,208,8,2), (112,224,8,2)] #,(112,224,8,2), (120,240,8,2), (128,256,8,2)
#
# params = [(16,48,6,2)]
# for p in params:
#     num_switches, num_servers, switch_k, num_servers_per_rack = p
#     switch_d = switch_k - num_servers_per_rack
#     file_name = 'StratAL'
#     s = Strat('C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d24/' + file_name + str(num_servers)+ "_"+str(num_servers_per_rack)+".txt")
#     G1 = s.G
#     write_graph_to_file(G1,'C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d24/' + file_name + str(num_servers)+ "_"+str(num_servers_per_rack)+"new.txt" )
def check_degree(G):
    d = list(G.degree())
    dd = []
    for i in d:
        dd.append(i[1])
    print('Max degree', max(dd))
    print('Min degree', min(dd))


from utilities import find_diameter, find_average_shortest_path, find_laplacian_spectrum, find_adjacency_spectrum
G = read_graph_from_file("C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d15/StratAL2048_8.txt")
print(find_average_shortest_path(G))
print(find_diameter(G))
x1, x2 = find_adjacency_spectrum(G)
print(x1-x2)
print(find_laplacian_spectrum(G))
check_degree(G)
print("################################################################")
G = read_graph_from_file("C:/Users/umroot/PycharmProjects/datacenter/new_xpander/d15/XpanderAL2048_8.txt")
print(find_average_shortest_path(G))
print(find_diameter(G))
x1, x2 = find_adjacency_spectrum(G)
print(x1-x2)
print(find_laplacian_spectrum(G))
check_degree(G)
print("################################################################")
G = read_graph_from_file("C:/Users/umroot/PycharmProjects/datacenter/new_jellyfish/d15/JellyfishAL2048_8.txt")
print(find_average_shortest_path(G))
print(find_diameter(G))
x1, x2 = find_adjacency_spectrum(G)
print(x1-x2)
print(find_laplacian_spectrum(G))
check_degree(G)


# file_name = 'StratAL'
# s = Strat('C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d24/StratAL48_2.txt')
# G1 = s.G
# write_graph_to_file(G1,"C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d15/StratAL4096_8_1.txt")