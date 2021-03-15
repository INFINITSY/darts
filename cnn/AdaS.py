"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini and Mathieu Tuli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import sys

from typing import List, Any
import numpy as np

from metrics import Metrics
from components import LRMetrics  # IOMetrics


class AdaS():
    n_buffer = 2

    def __init__(self, parameters: List[Any],
                 beta: float = 0.8, zeta: float = 1.,
                 p: int = 1, init_lr: float = 3e-2,
                 min_lr: float = 1e-20) -> None:
        '''
        parameters: list of torch.nn.Module.parameters()
        beta: float: AdaS gain factor [0, 1)
        eta: knowledge gain hyper-paramters [0, 1)
        init_lr: initial learning rate > 0
        min_lr: minimum possible learning rate > 0
        '''
        if beta < 0 or beta >= 1:
            raise ValueError
        # if zeta < 0 or zeta > 1:
        #     raise ValueError
        if init_lr <= 0:
            raise ValueError
        if min_lr <= 0:
            raise ValueError

        self.metrics = metrics = Metrics(parameters=parameters, p=p)
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.beta = beta
        self.zeta = zeta

        init_lr_vector = np.repeat(a=init_lr,
                                   repeats=len(metrics.layers_info))
        self.lr_vector = init_lr_vector
        self.velocity_moment_conv = np.zeros(metrics.number_of_all_conv)
        self.acceleration_moment_conv = np.zeros(metrics.number_of_all_conv)
        # without depth wise conv
        self.R_conv_normal = np.zeros(metrics.number_of_normal_conv)
        # with depth wise conv
        self.R_conv = np.zeros(metrics.number_of_all_conv)
        self.velocity_moment_fc = np.zeros(metrics.number_of_fc)[0]
        self.acceleration_moment_fc = np.zeros(metrics.number_of_fc)[0]
        self.R_fc = np.zeros(metrics.number_of_fc)[0]
        # whether to apply AdaS learning rate
        self.mask_conv_normal = np.zeros(metrics.number_of_normal_conv, dtype=bool)
        self.mask_conv = np.zeros(metrics.number_of_all_conv, dtype=bool)

    def step(self, epoch: int, metrics: Metrics = None) -> None:
        """
        velocity_conv_rank_normal_layer:
            velocity for each normal conv layer (no depth-wise),
            multiple values in one edge.

        velocity_conv_rank_normal_edge:
            velocity for each normal conv layer (no depth-wise),
            same values in one edge.

        velocity_conv_rank:
            velocity for each conv layer (including depth-wise),
            multiple values in one edge.

        self.R_conv_normal:
            learning rate for each normal conv layer (no depth-wise),
            multiple values in one edge.

        self.R_conv:
            learning rate for each conv layer (including depth-wise),
            multiple values in one edge.
        """
        if epoch == 0:
            velocity_conv_rank_normal_layer = np.zeros(len(self.metrics.normal_conv_indices))
            # velocity_fc_rank = self.init_lr * \
            #                    np.ones(len(self.metrics.fc_indices))[0]
            # NOTE unused (below)
            # acceleration_conv_rank = np.zeros(len(conv_indices))
            # acceleration_fc_rank = np.zeros(len(fc_indices))[0]
            # preserving_acceleration_conv = alpha
            # velocity_conv_rank = self.init_lr * np.ones(len(self.metrics.conv_all_indices))

            # without depth wise conv, one lr per layer
            self.R_conv_normal = self.init_lr * np.ones(len(self.metrics.normal_conv_indices))
            # with depth wise conv, one lr per layer
            self.R_conv = self.init_lr * np.ones(len(self.metrics.conv_all_indices))
            self.R_fc = self.init_lr * np.ones(len(self.metrics.fc_indices))[0]
        else:
            n_replica = AdaS.n_buffer - min(epoch + 1, AdaS.n_buffer)
            input_channel_replica = np.tile(
                A=metrics.historical_metrics[0].input_channel_S,
                reps=(n_replica, 1))
            output_channel_replica = np.tile(
                A=metrics.historical_metrics[0].output_channel_S,
                reps=(n_replica, 1))
            fc_channel_replica = np.tile(
                A=metrics.historical_metrics[0].fc_S, reps=(n_replica, 1))
            for iteration in range(AdaS.n_buffer - n_replica):
                epoch_identifier = (epoch - AdaS.n_buffer +
                                    n_replica + iteration + 1)
                metric = metrics.historical_metrics[epoch_identifier]  # last two epochs
                input_channel_replica = np.concatenate((
                    input_channel_replica,
                    np.tile(
                        A=metric.input_channel_S,
                        reps=(1, 1))))
                output_channel_replica = np.concatenate(
                    (output_channel_replica, np.tile(
                        A=metric.output_channel_S,
                        reps=(1, 1))))
                fc_channel_replica = np.concatenate(
                    (fc_channel_replica, np.tile(
                        A=metric.fc_S,
                        reps=(1, 1))))
            x_regression = np.linspace(start=0, stop=AdaS.n_buffer - 1,
                                       num=AdaS.n_buffer)

            channel_replica = (input_channel_replica +
                               output_channel_replica) / 2
            # channel_replica = output_channel_replica

            """Calculate Rank Velocity"""
            velocity_conv_rank_normal_layer = np.polyfit(
                x=x_regression, y=channel_replica, deg=1)[0]  # slope / rate of change of S for each conv layer
            velocity_fc_rank = np.polyfit(
                x=x_regression, y=fc_channel_replica, deg=1)[0][0]
            # start applying AdaS to a layer if it has non-zero S
            self.mask_conv_normal[(velocity_conv_rank_normal_layer > 0) & ~self.mask_conv_normal] = True

            ################################################################################
            # compute the avg velocity for each edge
            # edge_reduce_FR = sum([[i*(i+3)//2, i*(i+3)//2+1] for i in range(4)], [])
            # velocity_conv_rank_normal_edge = velocity_conv_rank_normal_layer.copy()  # velocity per edge
            # offset = 1  # first layer is stem
            # cells = 8
            # edges = 14
            # for i_cell in range(cells):
            #     offset += 3 if i_cell in [cells//3 + 1, 2*cells//3 + 1] else 2
            #     for i_edge in range(edges):
            #         if i_cell in [cells//3, 2*cells//3] and i_edge in edge_reduce_FR:
            #             # every first 2 edges of each node in reduction cell
            #             # [0, 1, 2, 3, 5, 6, 9, 10]
            #             # 8 conv layers in one edge
            #             mean_velocity = np.mean(velocity_conv_rank_normal_layer[offset: offset + 8])
            #             velocity_conv_rank_normal_edge[offset: offset + 8] = mean_velocity
            #             offset += 8
            #         else:
            #             # normal cell
            #             # or other edges in reduction cell [4, 7, 8, 11, 12, 13]
            #             # 6 conv layers in one edge
            #             mean_velocity = np.mean(velocity_conv_rank_normal_layer[offset: offset + 6])
            #             velocity_conv_rank_normal_edge[offset: offset + 6] = mean_velocity
            #             offset += 6
            #
            # include depth wise conv layers
            i_normal = 0
            velocity_conv_rank = np.zeros(len(self.metrics.conv_all_indices))
            for i in range(len(velocity_conv_rank)):
                velocity_conv_rank[i] = velocity_conv_rank_normal_layer[i_normal]
                self.mask_conv[i] = self.mask_conv_normal[i_normal]
                if self.metrics.conv_all_indices[i] == self.metrics.normal_conv_indices[i_normal]:
                    i_normal += 1
            ################################################################################

            # v  = dS / dt   ,   l' = beta * l + zeta * v     (0.8*l + v)
            # large velocity in S -> large lr
            # without depth wise conv, one lr per layer
            # self.R_conv_normal = self.beta * self.R_conv_normal + self.zeta * velocity_conv_rank_normal_layer
            self.R_conv_normal[self.mask_conv_normal] = self.beta * self.R_conv_normal[
                self.mask_conv_normal] + self.zeta * velocity_conv_rank_normal_layer[self.mask_conv_normal]
            # with depth wise conv, one lr per layer
            # self.R_conv = self.beta * self.R_conv + self.zeta * velocity_conv_rank
            self.R_conv[self.mask_conv] = self.beta * self.R_conv[self.mask_conv] + \
                                          self.zeta * velocity_conv_rank[self.mask_conv]
            self.R_fc = self.beta * self.R_fc + self.zeta * velocity_fc_rank

        self.R_conv_normal = np.maximum(self.R_conv_normal, self.min_lr)
        self.R_conv = np.maximum(self.R_conv, self.min_lr)
        self.R_fc = np.maximum(self.R_fc, self.min_lr)

        # call_indices_conv = np.concatenate(
        #     (self.metrics.conv_indices, [self.metrics.fc_indices[0]]), axis=0)
        call_indices_conv = np.concatenate(
            (self.metrics.conv_all_indices, [self.metrics.fc_indices[0]]), axis=0)
        for iteration_conv in range(len(call_indices_conv) - 1):
            index_start = call_indices_conv[iteration_conv]
            index_end = call_indices_conv[iteration_conv + 1]
            self.lr_vector[index_start: index_end] = \
                self.R_conv[iteration_conv]

        call_indices_fc = np.concatenate(
            (self.metrics.fc_indices,
             [len(self.metrics.layers_info)]), axis=0)
        for iteration_fc in range(len(call_indices_fc) - 1):
            index_start = call_indices_fc[iteration_fc]
            index_end = call_indices_fc[iteration_fc + 1]
            self.lr_vector[index_start: index_end] = self.R_fc
        # return LRMetrics(rank_velocity=velocity_conv_rank.tolist(),
        #                  r_conv=self.R_conv.tolist())
        return LRMetrics(rank_velocity=velocity_conv_rank_normal_layer.tolist(),
                         r_conv=self.R_conv_normal.tolist())
