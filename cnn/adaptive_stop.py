import numpy as np

conv_start_id_normal = [3, 89, 278, 364, 553, 639]
conv_start_id_reduce = [175, 450]
num_conv_per_edge_normal = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
num_conv_per_edge_reduce = [8, 8, 8, 8, 6, 8, 8, 6, 6, 8, 8, 6, 6, 6]

class StopChecker:

    def __init__(self) -> None:
        self.smooth_window_size = 5
        self.epsilon = 0
        self.conv_layers_index_stop = None
        self.normal_edge_index_stop = np.zeros(len(num_conv_per_edge_normal), dtype=bool)  # 14
        self.reduce_edge_index_stop = np.zeros(len(num_conv_per_edge_reduce), dtype=bool)  # 14

    # def local_stop(self, metrics: Metrics, epoch: int) -> None:
    def local_stop(self, input_channel_S, epoch: int):
        """Use the smoothed input knowledge gain to determine whether to stop
        searching on a certain edge.

        :param metrics:
        :param epoch:
        :return: None

        For each edge, if any of its average delta knowledge gain in each cell is
        below a threshold, we stop searching this edge.
        """
        # take last {window_size+1} cells for smoothing
        # input_channel_replica = np.tile(
        #     A=metrics.historical_metrics[0].input_channel_S,
        #     reps=(0, 1))
        # for iteration in range(self.smooth_window_size + 2):
        #     epoch_identifier = (epoch - self.smooth_window_size - 1 + iteration)
        #     metric = metrics.historical_metrics[epoch_identifier]  # last 4 epochs
        #     input_channel_replica = np.concatenate((
        #         input_channel_replica,
        #         np.tile(
        #             A=metric.input_channel_S,
        #             reps=(1, 1))))
        if self.conv_layers_index_stop is None:
            self.conv_layers_index_stop = np.zeros(input_channel_S.shape[1], dtype=bool)

        input_channel_replica = np.tile(
            A=input_channel_S[0, :],
            reps=(0, 1))
        for iteration in range(self.smooth_window_size + 2):
            epoch_identifier = (epoch - self.smooth_window_size - 1 + iteration)  # last 4 epochs
            input_channel_replica = np.concatenate((
                input_channel_replica,
                np.tile(
                    A=input_channel_S[epoch_identifier, :],
                    reps=(1, 1))))
        # padding for smoothing
        input_channel_replica = np.pad(input_channel_replica, ((0, 2), (0, 0)), 'edge')

        # smoothing
        num_layers = input_channel_replica.shape[1]
        for layer in range(num_layers):
            input_channel_replica[:, layer] = np.convolve(input_channel_replica[:, layer],
                                                          np.ones(self.smooth_window_size)/self.smooth_window_size,
                                                          mode='same')

        # since we repeat (padding) the last input_S for smoothing, now we remove it
        input_channel_replica = input_channel_replica[:-2, :]

        # stopping criterion 1: knowledge gain
        conv_layers_delta_S = input_channel_replica[-1, :]-input_channel_replica[-self.smooth_window_size, :]
        # conv_layers_index_stop = np.zeros_like(conv_layers_delta_S, dtype=bool)

        # edges in normal cell
        offset_start = 0
        offset_end = 0
        # Iterate through 14 edges in normal cells
        for edge in range(len(num_conv_per_edge_normal)):

            # check if the current edge is already stopped
            if self.normal_edge_index_stop[edge]:
                offset_end += num_conv_per_edge_normal[edge]
                offset_start += num_conv_per_edge_normal[edge]
                continue

            offset_end += num_conv_per_edge_normal[edge]
            # stop_flag = False
            stop_flag_cell = np.zeros(len(conv_start_id_normal), dtype=bool)

            # Compute the average of all conv layers in each edge and each cell
            for cell in range(len(conv_start_id_normal)):
                start = conv_start_id_normal[cell] + offset_start
                end = conv_start_id_normal[cell] + offset_end

                avg_delta_S = np.mean(conv_layers_delta_S[start: end])
                # conv_layers_delta_S[start: end] = avg_delta_S

                # If any avg_delta_S is below the threshold, stop_flag will be True
                # stop_flag = stop_flag | (avg_delta_S < self.epsilon)
                stop_flag_cell[cell] = (avg_delta_S < self.epsilon)

            # For each edge, if its delta_S in >=2 cells are below a threshold,
            # stop searching it
            stop_flag = np.sum(stop_flag_cell) >= 2
            self.normal_edge_index_stop[edge] = stop_flag
            for cell in range(len(conv_start_id_normal)):
                start = conv_start_id_normal[cell] + offset_start
                end = conv_start_id_normal[cell] + offset_end

                self.conv_layers_index_stop[start: end] = stop_flag

            offset_start += num_conv_per_edge_normal[edge]

        # edges in reduction cell
        offset_start = 0
        offset_end = 0
        # Iterate through 14 edges in reduction cells
        for edge in range(len(num_conv_per_edge_reduce)):

            # check if the current edge is already stopped
            if self.reduce_edge_index_stop[edge]:
                offset_end += num_conv_per_edge_reduce[edge]
                offset_start += num_conv_per_edge_reduce[edge]
                continue

            offset_end += num_conv_per_edge_reduce[edge]
            # stop_flag = False
            stop_flag_cell = np.zeros(len(conv_start_id_normal), dtype=bool)

            # Compute the average of all conv layers in each edge and each cell
            for cell in range(len(conv_start_id_reduce)):
                start = conv_start_id_reduce[cell] + offset_start
                end = conv_start_id_reduce[cell] + offset_end

                avg_delta_S = np.mean(conv_layers_delta_S[start: end])
                # conv_layers_delta_S[start: end] = avg_delta_S

                # If any avg_delta_S is below the threshold, stop_flag will be True
                # stop_flag = stop_flag | (avg_delta_S < self.epsilon)
                stop_flag_cell[cell] = (avg_delta_S < self.epsilon)

            # For each edge, if any of its avg delta_S in each cell is below a threshold,
            # stop searching it
            stop_flag = np.sum(stop_flag_cell) >= 2
            self.reduce_edge_index_stop[edge] = stop_flag
            for cell in range(len(conv_start_id_reduce)):
                start = conv_start_id_reduce[cell] + offset_start
                end = conv_start_id_reduce[cell] + offset_end

                self.conv_layers_index_stop[start: end] = stop_flag

            offset_start += num_conv_per_edge_reduce[edge]

        # Include depth wise conv layers,
        # and update layers_index_todo
        # i_conv = 0
        # for i_all_conv in range(len(metrics.conv_all_indices)):
        #     # Fetch the index of current conv layer in all layers (depth,point,fc...)
        #     conv_all_index = metrics.conv_all_indices[i_all_conv]
        #
        #     # If index_stop is True, then index_todo is False
        #     metrics.layers_index_todo[conv_all_index] = ~self.conv_layers_index_stop[i_conv]
        #     if metrics.conv_all_indices[i_all_conv] == metrics.normal_conv_indices[i_conv]:
        #         i_conv += 1

        return conv_layers_delta_S, self.conv_layers_index_stop, input_channel_replica

    def global_stop(self, H) -> None:
        pass







