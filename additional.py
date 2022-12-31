from routing import Routing
from utilities import read_graph_from_file

strat_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_strat/d24/'
xpander_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_xpander/d24/'
jellyfish_path = 'C:/Users/umroot/PycharmProjects/datacenter/ad_list_jellyfish/d24/'
G = read_graph_from_file(strat_path + 'StratAL2048_8.txt')
G1 = read_graph_from_file(xpander_path + 'XpanderAL2048_8.txt')
G2 = read_graph_from_file(jellyfish_path + 'JellyfishAL2048_8.txt')




def helper(x, k):
    r = Routing(x)
    paths = r.ksp(k)
    l = 0
    for i in x.nodes():
        for j in x.nodes():
            if i!=j:
                for k in paths[i][j]:
                    l += len(k) - 1
    N = len(x.nodes())
    return l/(N*(N-1))

def helper1(x, k):
    r = Routing(x)
    paths = r.ksp(k)
    l = 0
    for i in x.nodes():
        for j in x.nodes():
            if i!=j:
                l += len(paths[i][j][-1]) - 1
    N = len(x.nodes())
    return l/(N*(N-1))


s=[]
s1=[]
x=[]
x1=[]
j=[]
j1=[]
for i in range(1, 6):
    s.append(helper(G,i))
    s1.append(helper1(G,i))
    x.append(helper(G1,i))
    x1.append(helper1(G1,i))
    j.append(helper(G2,i))
    j1.append(helper1(G2,i))

print(s)
print(s1)
print(x)
print(x1)
print(j)
print(j1)
