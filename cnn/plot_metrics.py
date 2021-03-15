import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

metrics_file = '../tmp/DARTS_adas_c100/metrics_stat_adas_c100_stop_lr_05_beta_97_batch_56_time_60.xlsx'
# metrics_file = '~/MiLeNAS/save_data/milenas_adas_c100_05_9/metrics_stat_milenas_adas_c100_05_16_9_batch_64_probing_time_24.xlsx'
weights_file = '../tmp/DARTS_c100/weights_stat_c100_lr_05_batch_56_time_60.xlsx'
dfs = pd.read_excel(metrics_file, engine='openpyxl')

epoch_num = 24

# weights
# normal
# edges_num = 14
# normal_none = np.zeros((edges_num, epoch_num))
# normal_max = np.zeros((edges_num, epoch_num))
# normal_avg = np.zeros((edges_num, epoch_num))
# normal_skip = np.zeros((edges_num, epoch_num))
# normal_sep3 = np.zeros((edges_num, epoch_num))
# normal_sep5 = np.zeros((edges_num, epoch_num))
# normal_dil3 = np.zeros((edges_num, epoch_num))
# normal_dil5 = np.zeros((edges_num, epoch_num))
#
# for epoch in range(epoch_num):
#     normal_none[:, epoch] = dfs['normal_none_epoch'+str(epoch)]
#     normal_max[:, epoch] = dfs['normal_max_epoch'+str(epoch)]
#     normal_avg[:, epoch] = dfs['normal_avg_epoch'+str(epoch)]
#     normal_skip[:, epoch] = dfs['normal_skip_epoch'+str(epoch)]
#     normal_sep3[:, epoch] = dfs['normal_sep_3_epoch'+str(epoch)]
#     normal_sep5[:, epoch] = dfs['normal_sep_5_epoch'+str(epoch)]
#     normal_dil3[:, epoch] = dfs['normal_dil_3_epoch'+str(epoch)]
#     normal_dil5[:, epoch] = dfs['normal_dil_5_epoch'+str(epoch)]
#
# plt_pose = [1, 2, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19, 20]
# for i in range(edges_num):
#     plt.subplot(3, 5, i+1)
#     plt.title(f'edge_{i}')
#     plt.plot(range(epoch_num), normal_none[i, :], color='k', label='none')
#     plt.plot(range(epoch_num), normal_max[i, :], color='red', label='max')
#     plt.plot(range(epoch_num), normal_avg[i, :], color='orange', label='avg')
#     plt.plot(range(epoch_num), normal_skip[i, :], color='y', label='skip')
#     plt.plot(range(epoch_num), normal_sep3[i, :], color='c', label='sep_3')
#     plt.plot(range(epoch_num), normal_sep5[i, :], color='cyan', label='sep_5')
#     plt.plot(range(epoch_num), normal_dil3[i, :], color='purple', label='dil_3')
#     plt.plot(range(epoch_num), normal_dil5[i, :], color='violet', label='dil_5')
#
#
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
# plt.show()

# metrics
in_S = list()
out_S = list()
in_condition = list()
out_condition = list()

for epoch in range(epoch_num):
    in_S.append(dfs['in_S_epoch_' + str(epoch)].tolist())
    out_S.append(dfs['out_S_epoch_' + str(epoch)].tolist())
    in_condition.append(dfs['in_condition_epoch_' + str(epoch)].tolist())
    out_condition.append(dfs['out_condition_epoch_' + str(epoch)].tolist())

in_S = np.array(in_S)
out_S = np.array(out_S)
in_condition = np.array(in_condition)
out_condition = np.array(out_condition)

# plot data
metrics_data = in_S
metrics_name = 'S'

# cell_indices = [1, 171, 341, 527, 698, 868, 1054, 1225]

# colors = ['red', 'orangered', 'orange', 'gold', 'yellow', 'yellowgreen', 'greenyellow',
#           'lightgreen', 'turquoise', 'skyblue', 'cornflowerblue', 'blue', 'blueviolet', 'purple']
colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
          (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
          (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
          (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]
# for reduce cells
mask_1 = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
mask_2 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
mask_3 = [8, 8, 8, 8, 6, 8, 8, 6, 6, 8, 8, 6, 6, 6]
# mask_3 = [14, 14, 14, 14, 12, 14, 14, 12, 12, 14, 14, 12, 12, 12]
mask = [mask_1, mask_1, mask_1, mask_1, mask_2, mask_1, mask_1, mask_2, mask_2, mask_1, mask_1, mask_2, mask_2, mask_2]

# cell 0
layer_indices = range(3, 87)
# layer_indices = range(4, 171, 2)
layer = 0
edge = -1
plt.subplot(241)
for index in layer_indices:
    if layer % 6 == 0:
        edge += 1
    plt.plot(range(epoch_num), metrics_data[:, index], color=colors[edge])
    layer += 1

plt.title(f'Input {metrics_name} cell 0')
plt.ylim([0.0, 0.21])
plt.ylabel(f'{metrics_name}')

# cell 1
layer_indices = range(89, 172)
# layer_indices = range(174, 341, 2)
layer = 0
edge = -1
plt.subplot(242)
for index in layer_indices:
    if layer % 6 == 0:
        edge += 1
    plt.plot(range(epoch_num), metrics_data[:, index], color=colors[edge])
    layer += 1
plt.ylim([0.0, 0.21])
plt.title(f'Input {metrics_name} cell 1')

# cell 2 reduce
layer = 0
edge = 0
plt.subplot(243)
for index in range(175, 275):
# for index in range(343, 527):
    if layer == mask_3[edge]:
        edge += 1
        layer = 0
    # if mask[edge][layer] == 0:
    #     layer += 1
    #     continue
    plt.plot(range(epoch_num), metrics_data[:, index], color=colors[edge])
    layer += 1
plt.ylim([0.0, 0.21])
plt.title(f'Input {metrics_name} cell 2 (reduce)')

# cell 3
layer_indices = range(278, 362)
# layer_indices = range(531, 698, 2)
layer = 0
edge = -1
plt.subplot(244)
for index in layer_indices:
    if layer % 6 == 0:
        edge += 1
        plt.plot(range(epoch_num), metrics_data[:, index], color=colors[edge], label=f'edge {edge}')
    else:
        plt.plot(range(epoch_num), metrics_data[:, index], color=colors[edge])
    layer += 1
plt.ylim([0.0, 0.21])
plt.title(f'Input {metrics_name} cell 3')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

# cell 4
layer_indices = range(364, 448)
# layer_indices = range(701, 868, 2)
layer = 0
edge = -1
plt.subplot(245)
for index in layer_indices:
    if layer % 6 == 0:
        edge += 1
    plt.plot(range(epoch_num), metrics_data[:, index], color=colors[edge])
    layer += 1
plt.ylim([0.0, 0.21])
plt.title(f'Input {metrics_name} cell 4')
plt.xlabel('Epochs')
plt.ylabel(f'{metrics_name}')

# cell 5 reduce
layer = 0
edge = 0
plt.subplot(246)
for index in range(450, 550):
# for index in range(870, 1054):
    if layer == mask_3[edge]:
        edge += 1
        layer = 0
    # if mask[edge][layer] == 0:
    #     layer += 1
    #     continue
    plt.plot(range(epoch_num), metrics_data[:, index], color=colors[edge])
    layer += 1
plt.ylim([0.0, 0.21])
plt.title(f'Input {metrics_name} cell 5 (reduce)')
plt.xlabel('Epochs')

# cell 6
layer_indices = range(553, 637)
# layer_indices = range(1058, 1225, 2)
layer = 0
edge = -1
plt.subplot(247)
for index in layer_indices:
    if layer % 6 == 0:
        edge += 1
    plt.plot(range(epoch_num), metrics_data[:, index], color=colors[edge])
    layer += 1
plt.ylim([0.0, 0.21])
plt.title(f'Input {metrics_name} cell 6')
plt.xlabel('Epochs')

# cell 7
layer_indices = range(639, 723)
# layer_indices = range(1228, 1395, 2)
layer = 0
edge = -1
plt.subplot(248)
for index in layer_indices:
    if layer % 6 == 0:
        edge += 1
    plt.plot(range(epoch_num), metrics_data[:, index], color=colors[edge])
    layer += 1
plt.ylim([0.0, 0.21])
plt.title(f'Input {metrics_name} cell 7')
plt.xlabel('Epochs')


plt.show()
