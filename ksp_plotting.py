import json
from utilities import plot_path_lengths


def read_paths_from_file(filename):
    f = open(filename)
    data = json.load(f)
    return data


def plot_throughput(filename):
    data = read_paths_from_file(filename)
    data_to_be_plotted = {'strat':{},'xpander':{},'jellyfish':{}}
    for i, j in data.items():
        for k,v in j.items():
            data_to_be_plotted[i][int(k)] = float(v)/(int(k)*8*256*0.05)
    # plot_path_lengths(data_to_be_plotted, 'k6_throughput', '# flows per server', 'normalized throughput')
    for k, v in data_to_be_plotted.items():
        print(k)
        for k1, v1 in v.items():
            print(k1, v1)

def plot_number_of_blocked_connections(filename):
    data = read_paths_from_file(filename)
    data_to_be_plotted = {'strat':{},'xpander':{},'jellyfish':{}}
    for i, j in data.items():
        for k,v in j.items():
            data_to_be_plotted[i][int(k)] = float(v)
    # plot_path_lengths(data_to_be_plotted, 'k6_blocked_conn', '# flows per server', 'blocked connections (%)')
    for k, v in data_to_be_plotted.items():
        print(k)
        for k1, v1 in v.items():
            print(k1, v1)

def plot_av_num_of_hops_of_acc_connections(filename):
    data = read_paths_from_file(filename)
    data_to_be_plotted = {'strat':{},'xpander':{},'jellyfish':{}}
    for i, j in data.items():
        for k,v in j.items():
            data_to_be_plotted[i][int(k)] = float(v)
    # plot_path_lengths(data_to_be_plotted, 'k6_av_num_of_hops', '# flows per server', 'average number of hops')
    for k, v in data_to_be_plotted.items():
        print(k)
        for k1, v1 in v.items():
            print(k1, v1)

# plot_throughput('topos_alphas_without_distribution_throughput_k_6.json')
# print("##########################")
# plot_number_of_blocked_connections('topos_alphas_without_distribution_blocked_k_6.json')
# print("##########################")
# plot_av_num_of_hops_of_acc_connections('topos_alphas_without_distribution_avsp_k_6.json')

#######################################################################################################################################################################
# UNDER FAILURE
#######################################################################################################################################################################


def plot_throughput_under_failure(filenames, alpha):
    data_to_be_plotted_ = {'strat': {}, 'xpander': {}, 'jellyfish': {}}
    data_to_be_plotted_normalized = {'strat': {}, 'xpander': {}, 'jellyfish': {}}
    for filename in filenames:
        data = read_paths_from_file(filename)
        # rates = ["0.1", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.2"]
        # rates = ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1", "0.11", "0.12"]
        # rates = ['0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.1','0.11','0.12','0.13','0.14','0.15','0.16','0.17','0.18','0.19','0.2','0.21','0.22','0.23','0.24','0.25','0.26','0.27','0.28','0.29','0.3']
        rates = ['0.01', '0.03', '0.06', '0.09', '0.12', '0.15', '0.18', '0.21', '0.24', '0.27', '0.3']

        initials = []
        for k, v in data.items():
            initials.append(data[k]["initial"]["num_of_ac_c"])
        dict_data = dict()
        for i in range(len(rates)):
           j = 0
           l = []
           for k, v in data.items():
               num_of_b_c = data[k][rates[i]]["num_of_b_c"]
               total_num_of_ac_c = (initials[j] - num_of_b_c)
               l.append(total_num_of_ac_c)
               j += 1
           dict_data[float(rates[i])] = l
        throughput = []
        normalized_throughput = []
        for k, v in dict_data.items():
            throughput.append((sum(v)*alpha)/len(v))
            normalized_throughput.append(float(((sum(v) * alpha) / len(v)) / (96 * 256 * 0.05)))
        d1 = dict()
        d2 = dict()
        for i in range(len(throughput)):
            d1[float(rates[i])] = throughput[i]
            d2[float(rates[i])] = normalized_throughput[i]
        if 'xpander' in filename:
            data_to_be_plotted_['xpander'] = d1
            data_to_be_plotted_normalized['xpander'] = d2
        elif 'jellyfish' in filename:
            data_to_be_plotted_['jellyfish'] = d1
            data_to_be_plotted_normalized['jellyfish'] = d2
        elif 'strat' in filename:
            data_to_be_plotted_['strat'] = d1
            data_to_be_plotted_normalized['strat'] = d2

    # plot_path_lengths(data_to_be_plotted_, 'k6_throughput_under_failure', '# flows per server', 'throughput')
    # plot_path_lengths(data_to_be_plotted_normalized, 'k6_throughput_normalized__under_failure', '# flows per server', 'normalized throughput')
    for k, v in data_to_be_plotted_normalized.items():
        print(k)
        for k1, v1 in v.items():
            print(k1, v1)


def plot_blocked_conn_under_failure(filenames):
    data_to_be_plotted = {'strat': {}, 'xpander': {}, 'jellyfish': {}}
    # rates = ["0.1", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.2"]
    # rates = ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1", "0.11", "0.12"]
    # rates = ['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1', '0.11', '0.12', '0.13',
    #          '0.14', '0.15', '0.16', '0.17', '0.18', '0.19', '0.2', '0.21', '0.22', '0.23', '0.24', '0.25', '0.26',
    #          '0.27', '0.28', '0.29', '0.3']
    rates = ['0.01', '0.03', '0.06', '0.09', '0.12', '0.15', '0.18', '0.21', '0.24', '0.27','0.3']

    for filename in filenames:
        data = read_paths_from_file(filename)
        initials = []
        for k, v in data.items():
            initials.append(data[k]["initial"]["num_of_b_c"])
        dict_data = dict()
        for i in range(len(rates)):
           j = 0
           l = []
           for k, v in data.items():
               num_of_b_c = data[k][rates[i]]["num_of_b_c"]
               total_num_of_b_c = (initials[j] + num_of_b_c)
               l.append(total_num_of_b_c)
               j += 1
           dict_data[float(rates[i])] = l
        b = []
        for k, v in dict_data.items():
            b.append(sum(v)/len(v))
        d1 = dict()
        for i in range(len(b)):
            d1[float(rates[i])] = b[i]
        if 'xpander' in filename:
            data_to_be_plotted['xpander'] = d1
        elif 'jellyfish' in filename:
            data_to_be_plotted['jellyfish'] = d1
        elif 'strat' in filename:
            data_to_be_plotted['strat'] = d1
    # plot_path_lengths(data_to_be_plotted, 'k6_blocked_conn', '# flows per server', 'average number of blocked connections')
    for k, v in data_to_be_plotted.items():
        print(k)
        for k1, v1 in v.items():
            print(k1, v1)


def plot_av_num_of_hops_under_failure(filenames):
    data_to_be_plotted = {'strat': {}, 'xpander': {}, 'jellyfish': {}}
    # rates = ["0.1", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.2"]
    # rates = ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1", "0.11", "0.12"]
    # rates = ['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1', '0.11', '0.12', '0.13',
    #          '0.14', '0.15', '0.16', '0.17', '0.18', '0.19', '0.2', '0.21', '0.22', '0.23', '0.24', '0.25', '0.26',
    #          '0.27', '0.28', '0.29', '0.3']
    rates = ['0.01', '0.03', '0.06', '0.09', '0.12', '0.15', '0.18', '0.21', '0.24', '0.27','0.3']
    for filename in filenames:
        data = read_paths_from_file(filename)
        initials = []
        for k, v in data.items():
            initials.append(data[k]["initial"]["paths_of_accepted_connections"])
        dict_data = dict()
        for i in range(len(rates)):
           j = 0
           l = []
           ll = []
           for k, v in data.items():
               l1 = []
               for k1, v1 in initials[j].items():
                   if k1 in data[k][rates[i]]["paths_of_accepted_connections"]:
                        l1.append(len(data[k][rates[i]]["paths_of_accepted_connections"][k1]) - 1)
                   elif k1 in data[k][rates[i]]["blocked_connections_temp"]:
                       pass
                   else:
                       l1.append(len(v1) - 1)
               l.append(sum(l1))
               ll.append(len(l1))
               j += 1
           dict_data[float(rates[i])] = sum(l) / sum(ll)

        if 'xpander' in filename:
            data_to_be_plotted['xpander'] = dict_data
        elif 'jellyfish' in filename:
            data_to_be_plotted['jellyfish'] = dict_data
        elif 'strat' in filename:
            data_to_be_plotted['strat'] = dict_data
    # plot_path_lengths(data_to_be_plotted, 'k6_av_num_of_hops', '# flows per server', 'average number of hops')
    for k, v in data_to_be_plotted.items():
        print(k)
        for k1, v1 in v.items():
            print(k1, v1)

files = ['ksp_data_to_be_used_jellyfish96_v4.json', 'ksp_data_to_be_used_xpander96_v4.json', 'ksp_data_to_be_used_strat96_v4.json']
plot_throughput_under_failure(files, 0.05)
print("##########################################")
plot_blocked_conn_under_failure(files)
print("##########################################")
plot_av_num_of_hops_under_failure(files)
