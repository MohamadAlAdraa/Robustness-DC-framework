import matplotlib
import numpy as np

matplotlib.use('Agg')
import networkx as nx
import matplotlib as mtp
import matplotlib.pyplot as plt
import xpander
import xpander1
import jellyfish
import math
import statistics


#########################
#
# Graphs related functions
#
#########################
def write_graph_to_file(G, filename):
    nx.write_adjlist(G, filename)


def read_graph_from_file(filename):
    return nx.read_adjlist(filename)


def draw_graph(G, imagename):
    pos = nx.circular_layout(G)
    # plt.figure(figsize=(15, 15))
    nx.draw(G, pos)
    # if you need to label the nodes
    labels = dict(zip(G.nodes(), [str(x) for x in G.nodes()]))
    nx.draw_networkx_labels(G, pos, labels, font_color="whitesmoke")
    plt.draw()
    plt.savefig(imagename)
    plt.clf()


#########################
#
# Metrics to be used for comparison between graphs
#
#########################
def find_average_shortest_path(G):
    return nx.average_shortest_path_length(G)


def find_betweenness_centrality(G):
    b = nx.betweenness_centrality(G, normalized=False)
    bb = b.values()
    std_ = statistics.stdev(bb)
    av_bt = 0
    for i in b.values():
        av_bt += i
    b_max = max([i for i in b.values()])
    b_min = min([i for i in b.values()])
    return b_max, b_min, av_bt/len(b.values()), std_, b


def find_edge_betweenness_centrality(G):
    eb = nx.edge_betweenness_centrality(G, normalized=False)
    ebb = eb.values()
    std_ = statistics.stdev(ebb)
    av_ebt = 0
    for i in eb.values():
        av_ebt += i
    eb_max = max([i for i in eb.values()])
    eb_min = min([i for i in eb.values()])
    return eb_max, eb_min, av_ebt/len(eb.values()), std_, eb
    

def find_average_clustering(G):
    return nx.average_clustering(G)


def find_diameter(G):
    return nx.diameter(G)


def find_adjacency_spectrum(G):
    e = nx.adjacency_spectrum(G)
    flat = abs(e.flatten())
    flat.sort()
    largest = flat[-1]
    secondLargest = flat[-2]
    return largest, secondLargest


def jain_fairness_index(b):
    num = 0
    denum = 0
    for i in b.values():
        num += i
        denum += i**2
    jfi = num**2 / (len(b.values())*denum)
    return jfi


def find_std_av_shortest_path_length(G):
    x = nx.all_pairs_shortest_path_length(G)
    l = []
    for i in x:
        l.append(list(i[1].values()))
    a = [item for sublist in l for item in sublist]
    b = [i for i in a if i != 0]
    return statistics.stdev(b)


def find_laplacian_spectrum(G):
    e = nx.laplacian_spectrum(G)
    flat = e.flatten()
    flat.sort()
    secondSmallest = flat[1]
    return secondSmallest


def lb_av_shortest_path(N, r):
    k = 1
    R = float(N - 1)
    for j in range(1, N):
        tmpR = R - (r * math.pow(r - 1, j - 1))
        if tmpR >= 0:
            R = tmpR
        else:
            k = j
            break
    opt_d = 0.0
    for j in range(1, k):
        opt_d += j * r * math.pow(r - 1, j - 1)
    opt_d += k * R
    opt_d /= (N - 1)
    return opt_d


#########################
#
# Helper methods to generate Jellyfish and Xpander
#
#########################
def xpander_num_servers(num_servers, num_servers_per_rack, switch_d, lift_k):
    num_switches = int(math.ceil(num_servers/num_servers_per_rack))
    if num_switches <= switch_d:
        return None
    num_lifts = int(math.ceil(math.log(num_switches/(switch_d+1), lift_k)))
    num_switches = int((switch_d + 1) * math.pow(lift_k, num_lifts))
    num_servers = num_switches * num_servers_per_rack
    return num_servers


def get_the_best_xpander(number_of_switches, number_of_hosts, number_of_hosts_per_switch, d_regular, number_of_iterations=1):
    obj_ = None
    # highest_spectral_gap = 100
    for i in range(number_of_iterations):
        print('iter', i)
        obj = xpander.Xpander(number_of_switches, number_of_hosts, number_of_hosts_per_switch, d_regular)
        # x1, x2 = find_adjacency_spectrum(obj.incremental_graph)
        # if x2 < highest_spectral_gap:
        #     highest_spectral_gap = x2
        #     obj_ = obj
        obj_ = obj
    return obj_.incremental_graph


def get_the_best_xpander1(number_of_switches, number_of_hosts, number_of_hosts_per_switch, d_regular, number_of_iterations=1, number_of_lifts=1):
    obj_ = None
    highest_spectral_gap = 0
    for i in range(number_of_iterations):
        print('iter', i)
        obj = xpander1.Xpander(number_of_switches, number_of_hosts, number_of_hosts_per_switch, d_regular, liftIter=number_of_lifts)
        if obj.highest_spectral_gap > highest_spectral_gap:
            highest_spectral_gap = obj.highest_spectral_gap
            obj_ = obj
    return obj_.deterministic_lifted_graph


def get_the_best_jellyfish(number_of_switches, d_regular, number_of_iterations=1):
    optimal_obj = None
    highest_spectral_gap = 0
    for i in range(number_of_iterations):
        print(i)
        obj = jellyfish.Jellyfish(number_of_switches, d_regular)
        x1, x2 = find_adjacency_spectrum(obj.G)
        # x2 = find_average_shortest_path(obj.G)
        if (x1-x2) > highest_spectral_gap:
            highest_spectral_gap = x1 - x2
            optimal_obj = obj
    return optimal_obj.G

#########################
#
# Some functions for plotting the results
#
#########################


colors = {"Xpander" : "gray",
            "STRAT" : "blue",
          "Jellyfish": "orange",
          "Opt": "red",
          "Lower bound": "red"}

markers = {"Xpander" : "o",
             "STRAT" : "*",
            "Jellyfish": "d",
           "Opt":">",
           "Lower bound": ">"}
# colors = {"xpander" : "gray",
#             "strat" : "blue",
#           "jellyfish": "orange",
#           "opt": "red",
#           "Lower bound": "red"}
#
# markers = {"xpander" : "^",
#              "strat" : "d",
#             "jellyfish": "v",
#            "opt":">",
#            "Lower bound": ">"}


def plot_path_lengths(path_lengths, file_name, xTitle, yTitle):
    # csfont = {'fontname': 'Times New Roman', 'fontsize': 20}
    mtp.rc('font', family='Times New Roman')
    # csfont = {'fontname': 'Times New Roman'}
    mtp.rcParams['axes.linewidth'] = 1.5  # set the value globally
    # plt.rcParams.update({
    #     "font.family": "serif",
    #     "font.serif": "Times New Roman"
    # })
    plt.figure(figsize=(8, 6))
    for topo_type, data in path_lengths.items():
        num_servers = sorted([float(k) for k in data.keys()])
        path_lengths = [float(data[i]) for i in num_servers]
        if topo_type == 'strat':
            topo_type='STRAT'
        elif topo_type == 'xpander':
            topo_type='Xpander'
        elif topo_type == 'jellyfish':
            topo_type='Jellyfish'
        elif topo_type == 'opt':
            topo_type='Opt'
        elif topo_type == 'Lower bound':
            topo_type='Lower bound'
        plt.plot(num_servers, path_lengths, color=colors[topo_type], linestyle='-', marker=markers[topo_type], markersize=12, label=topo_type, linewidth=2.5)
    plt.xlabel(xlabel='Number of Switches', fontsize=22)
    plt.ylabel(ylabel=r'$\overline{L}_{sp}$', fontsize=22)
    plt.legend(loc="best", prop={'size': 18})
    plt.grid(color='k', lw=0.2)
    plt.tick_params(labelsize=18)
    # plt.axhline(lw=1.8)
    # plt.axvline(lw=1.8)
    plt.savefig(file_name)
    plt.close('all')
