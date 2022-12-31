import json
import numpy as np
from utilities import plot_path_lengths


f_p = 'C:/Users/umroot/PycharmProjects/datacenter/failing_figures/node/'
f_p1 = 'C:/Users/umroot/PycharmProjects/datacenter/failing_figures/edge/'
initial_values = {'strat':[2.0939334637964775, 0.42313604940709665], 'xpander':[2.260419214774951, 0.5352902075818825], 'jellyfish':[2.2586304427592956, 0.5344860077974662]}

def read_results_from_file(filename):
    f = open(filename)
    data = json.load(f)
    return data


# QLRM_nodef_ = read_results_from_file(f_p+'node_file_av_shortest_path_std_failing_512.json')
# QLRM_edgef_ = read_results_from_file(f_p1+'edge_file_av_shortest_path_std_failing_512.json')
#
#
# btw_nodef_node = read_results_from_file(f_p+'node_file_betw_failing_512.json')
# btw_nodef_edge = read_results_from_file(f_p+'node_file_edge_betw_failing_512.json')
# btw_edgef_node = read_results_from_file(f_p1+'edge_file_betw_failing_512.json')
# btw_edgef_edge = read_results_from_file(f_p1+'edge_file_edge_betw_failing_512.json')
#
#
# lap_nodef_ = read_results_from_file(f_p+'node_file_laplacian_failing_512.json')
# lap_edgef_ = read_results_from_file(f_p1+'edge_file_laplacian_failing_512.json')
# #
# sp_nodef_ = read_results_from_file(f_p+'node_file_adjacency_failing_512.json')
# sp_edgef_ = read_results_from_file(f_p1+'edge_file_adjacency_failing_512.json')
#
# laplacian = read_results_from_file(f_p+'node_file_laplacian1_failing_512.json')
# sp = read_results_from_file(f_p+'node_file_adjacency1_failing_512.json')


# def QLRM(d):
#     num = {'strat':{}, 'xpander':{}, 'jellyfish':{}}
#     for k1,k2 in d.items():
#         n = initial_values[k1][1]/initial_values[k1][0]
#         for f_r, v in k2.items():
#             q_f_r = 0
#             for i in v:
#                 x1 = i[1]/i[0]
#                 x2 = n/x1
#                 q_f_r += x2
#             num[k1][int(float(f_r))] = q_f_r/500
#     return num
#
# num1 = QLRM(QLRM_nodef_)
# num2 = QLRM(QLRM_edgef_)
#
# plot_path_lengths(num1, 'QLRM_NODE_F', 'fail rate (%)', 'QLRM')
# plot_path_lengths(num2, 'QLRM_EDGE_F', 'fail rate (%)', 'QLRM')


# def btw(d):
#     num = {'strat':{}, 'xpander':{}, 'jellyfish':{}}
#     num1 = {'strat':{}, 'xpander':{}, 'jellyfish':{}}
#     for k1,k2 in d.items():
#         for f_r, v in k2.items():
#             q_f_r = []
#             for i in v:
#                 x = i[0] - i[1]
#                 q_f_r.append(x)
#             num[k1][int(float(f_r))] = "{:.6f}".format(np.mean(q_f_r))
#             num1[k1][int(float(f_r))] = "{:.6f}".format(np.std(q_f_r))
#     return num, num1
# #
# mean_nodef_nodebetw, std_nodef_nodebetw = btw(btw_nodef_node)
# mean_nodef_edgebetw, std_nodef_edgebetw = btw(btw_nodef_edge)
#
# mean_edgef_nodebetw, std_edgef_nodebetw = btw(btw_edgef_node)
# mean_edgef_edgebetw, std_edgef_edgebetw = btw(btw_edgef_edge)
#
# print('mean_nodef_nodebetw', mean_nodef_nodebetw)
# print('std_nodef_nodebetw', std_nodef_nodebetw)
# print('mean_nodef_nodebetw', mean_nodef_nodebetw)
# print('std_nodef_edgebetw', std_nodef_edgebetw)
#
# print('mean_nodef_nodebetw', mean_edgef_nodebetw)
# print('std_edgef_nodebetw', std_edgef_nodebetw)
# print('mean_nodef_nodebetw', mean_edgef_edgebetw)
# print('std_edgef_edgebetw', std_edgef_edgebetw)


# def laplacian_sp(d):
#     num = {'strat':{}, 'xpander':{}, 'jellyfish':{}}
#     num1 = {'strat':{}, 'xpander':{}, 'jellyfish':{}}
#     for k1,k2 in d.items():
#         for f_r, v in k2.items():
#             num[k1][int(float(f_r))] = "{:.6f}".format(np.mean([float(i) for i in v]))
#             num1[k1][int(float(f_r))] = "{:.6f}".format(np.std([float(i) for i in v]))
#     return num, num1
#
# num1, num2 = laplacian_sp(sp_nodef_)
# num3, num4 = laplacian_sp(sp_edgef_)
#
# plot_path_lengths(num1, 'sp_nodef', 'fail rate (%)', 'Spectral gap')
# plot_path_lengths(num3, 'sp_edgef', 'fail rate (%)', 'Spectral gap')



# num = {'strat':{}, 'xpander':{}, 'jellyfish':{}}
# for k1,k2 in QLRM.items():
#     # n = initial_values[k1][1]/initial_values[k1][0]
#     for f_r, v in k2.items():
#         q_f_r = 0
#         for i in v:
#             # x1 = i[1]/i[0]
#             # x2 = n/x1
#             q_f_r += i[1]
#         num[k1][int(float(f_r))] = q_f_r/500
#
# print(num)
# plot_path_lengths(sp, 'spectral gap', 'Fail Rate (%)', 'sp')

# num = {'strat':{}, 'xpander':{}, 'jellyfish':{}}
# for k1, k2 in laplacian.items():
#     for f_r, v in k2.items():
#         num[k1][int(float(f_r))] = np.std([float(i) for i in v])
# plot_path_lengths(num, 'std lap', 'Fail Rate (%)', 'laplacian')
# print(num)


# num = {'strat':{}, 'xpander':{}, 'jellyfish':{}}
# for k1,k2 in btw.items():
#     for f_r, v in k2.items():
#         q_f_r = 0
#         for i in v:
#             x1 = i[0]-i[1]
#             q_f_r += x1
#         num[k1][int(float(f_r))] = q_f_r/500
#
# print(num)
# plot_path_lengths(num, 'btw edge fa', 'Fail Rate (%)', 'btw')