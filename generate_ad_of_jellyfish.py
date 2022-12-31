from utilities import write_graph_to_file, get_the_best_jellyfish


def check_degree(G):
    d = list(G.degree())
    dd = []
    for i in d:
        dd.append(i[1])
    print('Max degree', max(dd))
    print('Min degree', min(dd))

params = [(11,44,14,4)] #,(1024,8192,32,8),(2048,16384,32,8), ,(64,512,23,8),(128,1024,23,8),(256,2048,23,8),(512,4096,23,8),(22,88,14,4),(44,176,14,4),(88,352,14,4),(176,704,14,4),(352,1408,14,4)
jellyfish_path = 'C:/Users/umroot/PycharmProjects/datacenter/new_xpander/d10/'
# jellyfish_path = ''
#,(256,2048,32,8),(512,4096,32,8),(1024,8192,32,8)
for p in params:
    num_switches, num_servers, switch_k, num_servers_per_rack = p
    switch_d = switch_k - num_servers_per_rack
    print("##########################################################")
    print('Create Jellyfish with the following params:\n')
    print('switch ports', switch_k)
    print('switch degree', switch_d)
    print('num_switches', num_switches)
    print('num_servers', num_servers)
    print('num_servers_per_rack', num_servers_per_rack, '\n')

    # Jellyfish graph
    jellyfish_graph = get_the_best_jellyfish(num_switches, switch_d, 100)
    write_graph_to_file(jellyfish_graph, jellyfish_path + "JellyfishAL" + str(num_servers)+ "_" + str(switch_d) +".txt")
    print('Jellyfish ad_list file with', num_servers, 'servers and', num_switches, 'switches has been created')
    print('Graph validation: ')
    check_degree(jellyfish_graph)
    # print('Number of nodes', len(jellyfish_graph.nodes()))
    # print('Max degree', max(list(jellyfish_graph.degree())))
    # print('Min degree', min(list(jellyfish_graph.degree())))
    # print("##########################################################")
