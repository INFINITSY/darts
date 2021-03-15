import matplotlib.pyplot as plt
import numpy as np
# file1 = open('darts_train.sh-no_bn_lr_025_ch_16_e_0_eval_full_44514456.out', 'r')
# file2 = open('darts_train_2.sh-no_bn_lr_025_ch_16_e_5_eval_full_44514558.out', 'r')
# file3 = open('darts_train_3.sh-no_bn_lr_025_ch_16_e_10_eval_full_44514579.out', 'r')
# file4 = open('darts_train_4.sh-no_bn_lr_025_ch_16_e_15_eval_full_44514599.out', 'r')
# file5 = open('darts_train_5.sh-no_bn_lr_025_ch_16_e_20_eval_full_44514613.out', 'r')
# file6 = open('darts_train_6.sh-no_bn_lr_025_ch_16_e_25_eval_full_44514654.out', 'r')
# file7 = open('darts_train_7.sh-no_bn_lr_025_ch_16_e_30_eval_full_44514670.out', 'r')
# file8 = open('darts_train_8.sh-no_bn_lr_025_ch_16_e_35_eval_full_44514691.out', 'r')
# file9 = open('darts_train_9.sh-no_bn_lr_025_ch_16_e_40_eval_full_44514908.out', 'r')
# file10 = open('darts_train_10.sh-no_bn_lr_025_ch_16_e_45_eval_full_44515098.out', 'r')
# file11 = open('darts_train_11.sh-no_bn_lr_025_ch_16_e_49_eval_full_44515121.out', 'r')

file1 = open('./DARTS_c100_eval_outfiles/darts_train_c100.sh-c100_no_bn_lr_025_ch_16_e_0_eval_full_44795763.out', 'r')
file2 = open('./DARTS_c100_eval_outfiles/darts_train_c100_2.sh-c100_no_bn_lr_025_ch_16_e_5_eval_full_44795766.out', 'r')
file3 = open('./DARTS_c100_eval_outfiles/darts_train_c100_3.sh-c100_no_bn_lr_025_ch_16_e_10_eval_full_44842643.out', 'r')
file4 = open('./DARTS_c100_eval_outfiles/darts_train_c100_4.sh-c100_no_bn_lr_025_ch_16_e_15_eval_full_44842645.out', 'r')
file5 = open('./DARTS_c100_eval_outfiles/darts_train_c100_5.sh-c100_no_bn_lr_025_ch_16_e_20_eval_full_44842703.out', 'r')
file6 = open('./DARTS_c100_eval_outfiles/darts_train_c100_6.sh-c100_no_bn_lr_025_ch_16_e_25_eval_full_44842706.out', 'r')
file7 = open('./DARTS_c100_eval_outfiles/darts_train_c100_7.sh-c100_no_bn_lr_025_ch_16_e_30_eval_full_44842708.out', 'r')
file8 = open('./DARTS_c100_eval_outfiles/darts_train_c100_8.sh-c100_no_bn_lr_025_ch_16_e_35_eval_full_44842711.out', 'r')
file9 = open('./DARTS_c100_eval_outfiles/darts_train_c100_9.sh-c100_no_bn_lr_025_ch_16_e_40_eval_full_44842714.out', 'r')


errors = []
lens = []
eval_epoch = 0
for file in [file1, file2, file3, file4, file5, file6, file7, file8, file9]:
    error = []
    for line in file:
        if 'valid_acc' in line:
            acc = float(line.split(' ')[-1])
            # print(acc)
            error.append(100-acc)
    # accs = list(map(eval, accs))
    # plt.plot(range(len(error)),error,label=f'epoch_{eval_epoch}')
    eval_epoch += 5
    errors.append(error)
    lens.append(len(error))

# plt.yticks(np.arange(0,100,5))
# plt.legend()
# plt.ylim((80,95))
# plt.show()

min_len = min(lens)
print(min_len)
errors_compare = []
for error in errors:
    errors_compare.append(100-error[min_len-1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(0, 41, 5), errors_compare, '-*', label='DARTS',color='b')
ax2 = ax.twinx()
num_skip = [0,0,0,0,2,5,6,7,8]
ax2.plot(range(0, 41, 5), num_skip, '-o', label='DARTS',color='g')


file1 = open('./DARTS_adas_c100_eval_outfiles/darts_train_c100.sh-adas_c100_no_bn_lr_05_ch_16_beta_97_e_0_eval_full_44859846.out', 'r')
file2 = open('./DARTS_adas_c100_eval_outfiles/darts_train_c100_2.sh-adas_c100_no_bn_lr_05_ch_16_beta_97_e_5_eval_full_44859851.out', 'r')
file3 = open('./DARTS_adas_c100_eval_outfiles/darts_train_c100_3.sh-adas_c100_no_bn_lr_05_ch_16_beta_97_e_10_eval_full_44859857.out', 'r')
file4 = open('./DARTS_adas_c100_eval_outfiles/darts_train_c100_4.sh-adas_c100_no_bn_lr_05_ch_16_beta_97_e_15_eval_full_44859872.out', 'r')
file5 = open('./DARTS_adas_c100_eval_outfiles/darts_train_c100_5.sh-adas_c100_no_bn_lr_05_ch_16_beta_97_e_20_eval_full_44859875.out', 'r')
file6 = open('./DARTS_adas_c100_eval_outfiles/darts_train_c100_6.sh-adas_c100_no_bn_lr_05_ch_16_beta_97_e_25_eval_full_44859877.out', 'r')
file7 = open('./DARTS_adas_c100_eval_outfiles/darts_train_c100_7.sh-adas_c100_no_bn_lr_05_ch_16_beta_97_e_30_eval_full_44859880.out', 'r')
file8 = open('./DARTS_adas_c100_eval_outfiles/darts_train_c100_8.sh-adas_c100_no_bn_lr_05_ch_16_beta_97_e_35_eval_full_44897995.out', 'r')
file9 = open('./DARTS_adas_c100_eval_outfiles/darts_train_c100_9.sh-adas_c100_no_bn_lr_05_ch_16_beta_97_e_40_eval_full_44859887.out', 'r')
file10 = open('./DARTS_adas_c100_eval_outfiles/darts_train_c100_10.sh-adas_c100_no_bn_lr_05_ch_16_beta_97_e_45_eval_full_44859894.out', 'r')
file11 = open('./DARTS_adas_c100_eval_outfiles/darts_train_c100_11.sh-adas_c100_no_bn_lr_05_ch_16_beta_97_e_49_eval_full_44859898.out', 'r')


errors = []
lens = []
eval_epoch = 0
for file in [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10, file11]:
    error = []
    for line in file:
        if 'valid_acc' in line:
            acc = float(line.split(' ')[-1])
            # print(acc)
            error.append(100-acc)
    # accs = list(map(eval, accs))
    # plt.plot(range(len(error)),error,label=f'epoch_{eval_epoch}')
    eval_epoch += 5
    errors.append(error)
    lens.append(len(error))

# plt.yticks(np.arange(0,100,5))
# plt.legend()
# plt.ylim((80,95))
# plt.show()

min_len = min(lens)
print(min_len)
min_len = 414
errors_compare = []
for error in errors:
    errors_compare.append(100-error[min_len-1])

ax.plot(range(0, 51, 5), errors_compare, '--*', label='DARTS+Adas, lr: 0.05, beta: 0.97', color='royalblue')

num_skip = [0, 0, 0, 2, 4, 4, 4, 4, 4, 6, 8]
ax2.plot(range(0, 51, 5), num_skip, '--o', label='DARTS+Adas, lr: 0.05, beta: 0.97', color='limegreen')


file1 = open('../tmp/DARTS_adas_c100/darts_adas_train_05_9/darts_adas_train_05_9.sh-adas_c100_lr_05_ch_16_beta_9_e_0_eval_full_45303510.out', 'r')
file2 = open('../tmp/DARTS_adas_c100/darts_adas_train_05_9/darts_adas_train_05_9_2.sh-adas_c100_lr_05_ch_16_beta_9_e_5_eval_full_45303516.out', 'r')
file3 = open('../tmp/DARTS_adas_c100/darts_adas_train_05_9/darts_adas_train_05_9_3.sh-adas_c100_lr_05_ch_16_beta_9_e_10_eval_full_45303523.out', 'r')
file4 = open('../tmp/DARTS_adas_c100/darts_adas_train_05_9/darts_adas_train_05_9_4.sh-adas_c100_lr_05_ch_16_beta_9_e_15_eval_full_45303526.out', 'r')
file5 = open('../tmp/DARTS_adas_c100/darts_adas_train_05_9/darts_adas_train_05_9_5.sh-adas_c100_lr_05_ch_16_beta_9_e_20_eval_full_45303532.out', 'r')
file6 = open('../tmp/DARTS_adas_c100/darts_adas_train_05_9/darts_adas_train_05_9_6.sh-adas_c100_lr_05_ch_16_beta_9_e_25_eval_full_45303538.out', 'r')
# file7 = open('../tmp/DARTS_adas_c100/darts_adas_train_05_9/darts_train_c100_7.sh-adas_c100_no_bn_lr_05_ch_16_beta_97_e_30_eval_full_44859880.out', 'r')
file8 = open('../tmp/DARTS_adas_c100/darts_adas_train_05_9/darts_adas_train_05_9_8.sh-adas_c100_lr_05_ch_16_beta_9_e_35_eval_full_45303559.out', 'r')
file9 = open('../tmp/DARTS_adas_c100/darts_adas_train_05_9/darts_adas_train_05_9_9.sh-adas_c100_lr_05_ch_16_beta_9_e_40_eval_full_45303564.out', 'r')
file10 = open('../tmp/DARTS_adas_c100/darts_adas_train_05_9/darts_adas_train_05_9_10.sh-adas_c100_lr_05_ch_16_beta_9_e_45_eval_full_45303582.out', 'r')
file11 = open('../tmp/DARTS_adas_c100/darts_adas_train_05_9/darts_adas_train_05_9_11.sh-adas_c100_lr_05_ch_16_beta_9_e_49_eval_full_45303588.out', 'r')

# file1 = open('../../MiLeNAS/save_data/train_c100/milenas_train.py-c100_default_e_0_eval_45249360.out', 'r')
# file2 = open('../../MiLeNAS/save_data/train_c100/milenas_train_2.py-c100_default_e_5_eval_45249362.out', 'r')
# file3 = open('../../MiLeNAS/save_data/train_c100/milenas_train_3.py-c100_default_e_10_eval_45249364.out', 'r')
# file4 = open('../../MiLeNAS/save_data/train_c100/milenas_train_4.py-c100_default_e_15_eval_45249365.out', 'r')
# file5 = open('../../MiLeNAS/save_data/train_c100/milenas_train_5.py-c100_default_e_20_eval_45249366.out', 'r')
# # file6 = open('~/MiLeNAS/save_data/train_c100/milenas_train.py-c100_default_e_25_eval_45249360.out', 'r')
# file7 = open('../../MiLeNAS/save_data/train_c100/milenas_train_7.py-c100_default_e_30_eval_45249369.out', 'r')
# file8 = open('../../MiLeNAS/save_data/train_c100/milenas_train_8.py-c100_default_e_35_eval_45249370.out', 'r')
# file9 = open('../../MiLeNAS/save_data/train_c100/milenas_train_9.py-c100_default_e_40_eval_45249371.out', 'r')
# file10 = open('../../MiLeNAS/save_data/train_c100/milenas_train_10.py-c100_default_e_45_eval_45249372.out', 'r')
# file11 = open('../../MiLeNAS/save_data/train_c100/milenas_train_11.py-c100_default_e_49_eval_45249373.out', 'r')


errors = []
lens = []
eval_epoch = 0
for file in [file1, file2, file3, file4, file5, file6, file8, file9, file10, file11]:
    error = []
    for line in file:
        if 'valid_acc' in line:
            acc = float(line.split(' ')[-1])
            # print(acc)
            error.append(100-acc)
    # accs = list(map(eval, accs))
    # plt.plot(range(len(error)),error,label=f'epoch_{eval_epoch}')
    eval_epoch += 5
    errors.append(error)
    lens.append(len(error))

# plt.yticks(np.arange(0,100,5))
# plt.legend()
# plt.ylim((80,95))
# plt.show()

min_len = min(lens)
print(min_len)
min_len = 414
errors_compare = []
for error in errors:
    if len(errors_compare) == 6:
        errors_compare.append(errors_compare[-1])
    errors_compare.append(100-error[min_len-1])

ax.plot(range(0, 51, 5), errors_compare, '--*', label='DARTS+Adas, lr: 0.05, beta: 0.9', color='lightskyblue')

num_skip = [0, 0, 1, 4, 5, 6, 6, 7, 7, 7, 7]
ax2.plot(range(0, 51, 5), num_skip, '--o', label='DARTS+Adas, lr: 0.05, beta: 0.9', color='palegreen')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test accuracy',color='b')
ax2.set_ylabel('#Skip-connection',color='g')
ax.set_title('CIFAR-100')

ax.legend(bbox_to_anchor=(0, 1), loc=3)
ax2.legend(bbox_to_anchor=(1, 1), loc=4)
plt.show()
