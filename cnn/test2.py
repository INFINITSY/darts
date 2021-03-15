import matplotlib.pyplot as plt
import numpy as np

# darts_025 = [0, 0, 0, 0, 2, 5, 6, 7, 8]
darts_025 = [0, 0, 0, 2, 3, 5, 7, 8]
darts_05 = [0, 0, 3, 3, 4, 4, 5, 7, 7]
adas_025_9 = [0, 0, 0, 0, 3, 5, 7]
adas_05_9 = [0, 0, 1, 4, 5, 6, 6, 7, 7, 7, 7]
adas_05_95 = []
adas_05_97 = [0, 0, 0, 2, 4, 4, 4, 4, 4, 6, 8]

mile = [0, 0, 0, 2, 4, 4, 4, 3, 4, 4, 4]
mile_adas_025_9 = [0, 0, 0, 0, 3, 4, 5, 5, 6, 6, 6]
mile_adas_05_9 = [0, 0, 0, 3, 4, 5, 5, 5, 5, 6, 6]
mile_adas_05_95 = [0, 0, 0, 0, 1, 1, 5, 5, 6, 6, 6]
mile_adas_05_97 = [0, 0, 0, 0, 0, 3, 3, 4, 4, 4, 4]

plt.plot(range(0, 36, 5), darts_025, '-o', label='DARTS, lr: 0.025')
# plt.plot(range(0, 41, 5), darts_05, '-o', label='DARTS, lr: 0.05')
#
# # plt.plot(range(0, 31, 5), adas_025_9, '-o', label='DARTS+Adas, lr: 0.025, beta: 0.9')
# # plt.plot(range(0, 51, 5), adas_05_9, '-o', label='DARTS+Adas, lr: 0.05, beta: 0.9')
# # plt.plot(range(0, 51, 5), adas_05_97, '-o', label='DARTS+Adas, lr: 0.05, beta: 0.97')
plt.plot(range(0, 51, 5), mile, '--o', label='MiLeNAS, lr: 0.025')
plt.plot(range(0, 51, 5), mile_adas_025_9, '--o', label='MiLeNAS+Adas, lr: 0.025, beta: 0.9')
plt.plot(range(0, 51, 5), mile_adas_05_9, '--o', label='MiLeNAS+Adas, lr: 0.05, beta: 0.9')
plt.plot(range(0, 51, 5), mile_adas_05_95, '--o', label='MiLeNAS+Adas, lr: 0.05, beta: 0.95')
plt.plot(range(0, 51, 5), mile_adas_05_97, '--o', linewidth=3.0, label='MiLeNAS+Adas, lr: 0.05, beta: 0.97')

plt.xlabel('Epoch')
plt.ylabel('#Skip-connection')
plt.legend()
plt.show()
