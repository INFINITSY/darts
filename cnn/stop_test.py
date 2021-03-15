from adaptive_stop import StopChecker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

metrics_file = './metrics_data/metrics_stat_adas_c100_no_bn_lr_05_ch_16_beta_97_epoch_50.xlsx'
df = pd.read_excel(metrics_file, engine='openpyxl')



stop = StopChecker()

in_S = list()

epoch_num = 50
for epoch in range(epoch_num):
    in_S.append(df['in_S_epoch_' + str(epoch)].tolist())

in_S = np.array(in_S)

# colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
#           (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
#           (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
#           (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]
colors = ['red', 'orangered', 'orange', 'gold', 'yellow', 'yellowgreen']

conv_start_id_normal = [3, 89, 278, 364, 553, 639]
conv_start_id_reduce = [175, 450]
num_conv_per_edge_normal = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
num_conv_per_edge_reduce = [8, 8, 8, 8, 6, 8, 8, 6, 6, 8, 8, 6, 6, 6]


start = 278+30
end = start+6
stop_flag = False
# plt.figure()
stops = np.zeros(14)
norm_edge = [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81]
for e in range(epoch_num):
    in_S_e = in_S[:e+1, :]
    # layer_indices = range(start, end)
    # i = 0
    # for index in layer_indices:
    #     plt.scatter(e, in_S_e[e, index], color=colors[i])
    #     i += 1
    # plt.pause(0.001)

    if e > 10:
        delta_S, index_stop, smooth_S_e = stop.local_stop(in_S_e, e)
        print('### ', e)
        # print(np.mean(delta_S[start: end]))
        #
        # print(index_stop[start])

        for i in range(14):
            if index_stop[norm_edge[i]] & (stops[i] == 0):
                stops[i] = e
        print(stops)
        # if index_stop[start]:
            # stop_flag = True
            # plt.vlines(e, 0, 0.21, colors="c", linestyles="dashed")

        # print('e:', e)
        # # print(smooth_S_e[-1, start: end], smooth_S_e[-5, start: end])
        # print(delta_S[start: end])
        # print(index_stop[start: end])
        # layer_indices = range(start, end)
        # j = 0
        # for index in layer_indices:
        #     plt.scatter(e, smooth_S_e[-1, index], marker='*', color=colors[j])
        #     j += 1
#
for edge in range(14):
    plt.figure()
    for cell in range(6):
        start = conv_start_id_normal[cell] + edge * 6
        end = start + 6
        layer_indices = range(start, end)
        for index in layer_indices:
            plt.subplot(2, 3, cell+1)
            plt.plot(range(epoch_num), in_S[:, index])
            plt.vlines(stops[edge], 0, 0.21, colors="c", linestyles="dashed")


# plt.vlines(stop_e, 0, 0.21, colors="c", linestyles="dashed")
plt.ylim([0.0, 0.21])
plt.show()