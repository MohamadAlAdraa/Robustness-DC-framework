# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
import math


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


def throughput_upper_bound(N, d, number_of_servers_per_rack):
    av_lb = lb_av_shortest_path(N, d)
    f = number_of_servers_per_rack * (N * (N - 1))
    th_up = (N * d) / (f * av_lb)
    return th_up


print(lb_av_shortest_path(16, 5))
print(throughput_upper_bound(16, 5, 1))