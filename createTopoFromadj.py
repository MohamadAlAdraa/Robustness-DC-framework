import networkx as nx

def check_degree(G):
    d = list(G.degree())
    dd = []
    for i in d:
        dd.append(i[1])
    print('Max degree', max(dd))
    print('Min degree', min(dd))

class Topo:
    G = nx.Graph()

    def __init__(self, filename):
        self.filename = filename
        self.G = nx.Graph()

        f = open(filename, "r")
        l = f.readlines()
        f.close()

        for i in range(len(l)):
            self.G.add_node(i)

        for i in range(len(l)):
            x = l[i].split("    ")
            for j in x:
                j = j.strip('\n')
                if int(j) < i:
                    self.G.add_edge(i, int(j))

topo = Topo("C:/Users/umroot/PycharmProjects/datacenter/256.txt")
network = topo.G
mapp = dict()
for i in range(256):
    mapp[i] = i+1

network1 = nx.relabel_nodes(network, mapp)

print(list(network.nodes))
check_degree(network)
print(nx.diameter(network))
print(nx.average_shortest_path_length(network))

print(list(nx.neighbors(network, 0)))
# nx.write_adjlist(network, "C:/Users/umroot/PycharmProjects/datacenter/256NX.txt")

print(list(network1.nodes))
check_degree(network1)
print(nx.diameter(network1))
print(nx.average_shortest_path_length(network1))

print(list(nx.neighbors(network1, 1)))
nx.write_adjlist(network1, "C:/Users/umroot/PycharmProjects/datacenter/256NX.txt")

fwd = '['
for i in range(1, 257):
    l = list(nx.neighbors(network1, i))
    s = ''
    s1 = ''
    for j in l:
        s += str(j) + ' '
        s1 += str(j) + ' '
    s += '\n'
    if i < 256:
        s1 += ';'
    else:
        s1 += ']'
    fwd += s1
    f = open("C:/Users/umroot/PycharmProjects/datacenter/256MY.txt", 'a')
    f.write(s)
    f.close()

f = open("C:/Users/umroot/PycharmProjects/datacenter/256FWD.txt", 'w')
f.write(fwd)
f.close()