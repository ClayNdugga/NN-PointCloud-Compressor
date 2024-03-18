
import tensorflow as tf
import numpy as np
import io
import os

from tensorflow.keras import callbacks
import matplotlib.pyplot as plt



################################# Training Callbacks #################################

class ReconstructionVisualizationCallback(callbacks.Callback):
    def __init__(self, log_dir, input_tensor, freq=5):
        super().__init__()
        self.log_dir = log_dir
        self.input_tensor = input_tensor  
        self.freq = freq  

    def on_epoch_end(self, epoch, logs=None):
        # Log reconstruction every 'freq' epochs
        if epoch % self.freq == 0:
            _ , reconstruction  = self.model(self.input_tensor, training=False)
            
            original = self.input_tensor[0].numpy()  
            reconstructed = reconstruction[0].numpy()  

            # Use the updated plotting function to create the figure
            fig = self.plot_pointcloud(original, reconstructed)

            # Save the plot to a buffer (in memory file)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)

            # Log the buffer to TensorBoard
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)  # Add batch dimension

            with tf.summary.create_file_writer(self.log_dir).as_default():
                tf.summary.image("Reconstruction", image, step=epoch)

    def plot_pointcloud(self, original_pc, reconstructed_pc):
        fig = plt.figure(figsize=(15, 10))
        views = [(20, 30), (30, 120), (40, 210)] 
        for i in range(3):
            ax1 = fig.add_subplot(2, 3, i + 1, projection='3d')
            ax1.scatter(original_pc[:, 0], original_pc[:, 2], original_pc[:, 1], s=0.7)
            ax1.view_init(elev=views[i][0], azim=views[i][1])
            ax1.set_title('Original')
            ax1.set_xlabel('X Label')
            ax1.set_ylabel('Y Label')
            ax1.set_zlabel('Z Label')

            # Plot reconstructed point cloud
            ax2 = fig.add_subplot(2, 3, i + 4, projection='3d')
            ax2.scatter(reconstructed_pc[:, 0], reconstructed_pc[:, 2], reconstructed_pc[:, 1], s=0.7)
            ax2.view_init(elev=views[i][0], azim=views[i][1])
            ax2.set_title('Reconstructed')
            ax2.set_xlabel('X Label')
            ax2.set_ylabel('Y Label')
            ax2.set_zlabel('Z Label')

        plt.tight_layout()
        return fig

class SaveReconstructionsCallback(callbacks.Callback):
    def __init__(self, input_tensor, save_freq=10, save_path='/home/user7/NTC Project/reconstruction_callback'):
        super(SaveReconstructionsCallback, self).__init__()
        self.input_tensor = input_tensor
        self.save_freq = save_freq
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
    def on_train_begin(self, logs=None):
        np.save(os.path.join(self.save_path, 'original_pointcloud.npy'), self.input_tensor.numpy())

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            reconstructed = self.model(self.input_tensor, training=False)
            np.save(os.path.join(self.save_path, f'reconstruction_epoch_{epoch}.npy'), reconstructed.numpy())


class AlphaSchedulerCallback(callbacks.Callback):
    def __init__(self, alpha_variable, boundaries, values):
        super().__init__()
        self.alpha_variable = alpha_variable
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
        tf.keras.backend.set_value(self.alpha_variable, new_alpha)








# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Training Script")
#     parser.add_argument(
#         "-c",
#         "--config",
#         type=str,
#         default="default",
#         help="config file to use",
#     )
#     args = parser.parse_args()

#     current_script_path = Path(__file__).parent.absolute()
#     config_path = current_script_path.parent / "configs" / f"{args.config}.yaml"

#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)

#     return config
