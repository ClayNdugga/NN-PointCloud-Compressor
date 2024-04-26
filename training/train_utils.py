import tensorflow as tf
import numpy as np
import io
import os

from tensorflow.keras import callbacks
from tensorboard.plugins.mesh import summary_v2 as mesh_summary
import matplotlib.pyplot as plt


################################# Training Callbacks #################################


class UpdateMuVarianceCallback(callbacks.Callback):
    def __init__(self, train_dataset, update_freq=5, begin_epoch=20):
        super().__init__()
        self.train_dataset = train_dataset
        self.update_freq = update_freq
        self.begin = begin_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.update_freq == (self.update_freq - 1) and epoch > self.begin:
            mu_accumulator = None
            variance_accumulator = None
            batch_count = 0

            for data in self.train_dataset:
                # Forward pass to get quantized data
                _, quantized_data = self.model(data, training=False)

                if mu_accumulator is None:
                    mu_accumulator = tf.zeros_like(quantized_data)
                    mu_accumulator = tf.reduce_mean(mu_accumulator, axis=0)

                    variance_accumulator = tf.zeros_like(quantized_data)
                    variance_accumulator = tf.reduce_mean(variance_accumulator, axis=0)

                mu_accumulator += tf.reduce_mean(quantized_data, axis=0)
                variance_accumulator += tf.math.reduce_variance(quantized_data, axis=0)
                batch_count += 1

            # Calculate mean and variance across all batches
            self.model.mu.assign(mu_accumulator / batch_count)
            self.model.variance.assign(variance_accumulator / batch_count)


class LogMuVarianceCallback(callbacks.Callback):
    def __init__(self, log_dir, update_freq=4):
        super(LogMuVarianceCallback, self).__init__()
        self.update_freq = update_freq
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.update_freq == (self.update_freq - 1):
            with self.writer.as_default():
                tf.summary.histogram("mu", self.model.mu, step=epoch)
                tf.summary.histogram("variance", self.model.variance, step=epoch)
                self.writer.flush()


class ReconstructionVisualizationCallback(callbacks.Callback):
    def __init__(self, log_dir, input_tensor, freq=5):
        super().__init__()
        self.log_dir = log_dir
        self.input_tensor = input_tensor
        self.freq = freq

    def on_epoch_end(self, epoch):
        if epoch % self.freq == 0:
            (_, reconstruction), _ = self.model(self.input_tensor, training=False)

            original = self.input_tensor[0].numpy()
            reconstructed = reconstruction[0].numpy()

            fig = self.plot_pointcloud(original, reconstructed)
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)

            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)  # Add batch dimension

            with tf.summary.create_file_writer(self.log_dir).as_default():
                tf.summary.image("Reconstruction", image, step=epoch)

    def plot_pointcloud(self, original_pc, reconstructed_pc):
        fig = plt.figure(figsize=(15, 10))
        views = [(20, 30), (30, 120), (40, 210)]
        for i in range(3):
            ax1 = fig.add_subplot(2, 3, i + 1, projection="3d")
            ax1.scatter(original_pc[:, 0], original_pc[:, 2], original_pc[:, 1], s=0.7)
            ax1.view_init(elev=views[i][0], azim=views[i][1])
            ax1.set_title("Original")
            ax1.set_xlabel("X Label")
            ax1.set_ylabel("Y Label")
            ax1.set_zlabel("Z Label")

            # Plot reconstructed point cloud
            ax2 = fig.add_subplot(2, 3, i + 4, projection="3d")
            ax2.scatter(
                reconstructed_pc[:, 0],
                reconstructed_pc[:, 2],
                reconstructed_pc[:, 1],
                s=0.7,
            )
            ax2.view_init(elev=views[i][0], azim=views[i][1])
            ax2.set_title("Reconstructed")
            ax2.set_xlabel("X Label")
            ax2.set_ylabel("Y Label")
            ax2.set_zlabel("Z Label")

        plt.tight_layout()
        return fig


class AlphaSchedulerCallback(callbacks.Callback):
    def __init__(self, boundaries, values):
        super().__init__()
        self.boundaries = boundaries
        self.values = values

    def on_epoch_begin(self, epoch, logs=None):
        # Update alpha based on the current epoch
        new_alpha = self.values[0]
        for boundary, value in zip(self.boundaries, self.values[1:]):
            if epoch >= boundary:
                new_alpha = value
            else:
                break
        self.model.alpha.assign(new_alpha)


class ReconstructionMeshVisualizationCallback(callbacks.Callback):
    def __init__(self, log_dir, input_tensor, freq=5):
        super().__init__()
        self.log_dir = log_dir
        self.input_tensor = input_tensor
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            (_, reconstruction), _ = self.model(self.input_tensor, training=False)

            writer = tf.summary.create_file_writer(self.log_dir)
            with writer.as_default():
                mesh_summary.mesh(
                    "Original",
                    vertices=self.input_tensor,
                    step=epoch,
                    config_dict={"colors": "green"},
                )
                mesh_summary.mesh(
                    "Reconstructed",
                    vertices=reconstruction,
                    step=epoch,
                    config_dict={"colors": "blue"},
                )

            writer.flush()
