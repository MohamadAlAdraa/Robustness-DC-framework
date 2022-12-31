from utilities import write_graph_to_file, get_the_best_xpander, get_the_best_xpander1


def check_degree(G):
    d = list(G.degree())
    dd = []
    for i in d:
        dd.append(i[1])
    print('Max degree', max(dd))
    print('Min degree', min(dd))

params = [(22,88,14,4),(44,176,14,4),(88,352,14,4),(176,704,14,4),(352,1408,14,4)] #, (32,256,23,8),(64,512,23,8),(128,1024,23,8)
# jellyfish_path = 'C:/Users/umroot/PycharmProjects/datacenter/new_xpander/d15/'
# params = [(16,32,8,2), (24,48,8,2), (32,64,8,2), (40,80,8,2), (48,96,8,2), (56,112,8,2), (64,128,8,2), (72,144,8,2), (80,160,8,2), (88,176,8,2), (96,192,8,2), (104,208,8,2),(112,224,8,2), (120,240,8,2), (128,256,8,2)]
xpander_path_a = 'C:/Users/umroot/PycharmProjects/datacenter/new_xpander/d10/'

for p in params:
    num_switches, num_servers, switch_k, num_servers_per_rack = p
    switch_d = switch_k - num_servers_per_rack
    print("##########################################################")
    print('Create Xpander with the following params:\n')
    print('switch ports', switch_k)
    print('switch degree', switch_d)
    print('num_switches', num_switches)
    print('num_servers', num_servers)
    print('num_servers_per_rack', num_servers_per_rack, '\n')

    # Xpander graph
    xpander_graph = get_the_best_xpander1(num_switches, num_servers, num_servers_per_rack, switch_d, 5)
    #nx.draw_circular(xpander_graph, with_labels= True)
    #plt.savefig('topooo')
    write_graph_to_file(xpander_graph, xpander_path_a+"XpanderAL"+str(num_servers)+ "_" + str(switch_d) +".txt")
    print('Xpander ad_list file with', num_servers, 'servers and', num_switches, 'switches has been created')
    print('Graph validation: ')
    check_degree(xpander_graph)
