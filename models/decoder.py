import tensorflow as tf
from tensorflow.keras import layers


class MLP(layers.Layer):
    def __init__(self, layer_dims, batch_normalization = False):
        super(MLP, self).__init__()
        self.fully_connected_layers = [layers.Dense(layer_dim) for layer_dim in layer_dims[:-1]]
        self.batch_layers = [layers.BatchNormalization() for _ in layer_dims[:-1]]
        self.final_dense = layers.Dense(layer_dims[-1])
        self.batch_normalization = batch_normalization

    def call(self, x):
        for dense, batch_norm in zip(self.fully_connected_layers, self.batch_layers):
            x = dense(x)
            if self.batch_normalization:
                x = batch_norm(x)
            # x = tf.nn.relu(x)
        x = self.final_dense(x)
        return x

class MLP_conv(layers.Layer):
    def __init__(self, layer_dims, batch_normalization = False):
        super(MLP_conv, self).__init__()
        self.conv_layers = [layers.Conv1D(layer_dim, kernel_size = 1) for layer_dim in layer_dims[:-1]]
        self.batch_layers = [layers.BatchNormalization() for _ in layer_dims[:-1]]
        self.final_conv = layers.Conv1D(layer_dims[-1], kernel_size = 1)
        self.batch_normalization = batch_normalization

    def call(self, x):
        for conv, batch_norm in zip(self.conv_layers, self.batch_layers):
            x = conv(x)
            if self.batch_normalization:
                x = batch_norm(x)
            x = tf.nn.relu(x)
        x = self.final_conv(x)
        return x
    

class Decoder(layers.Layer):
    def __init__(self, coarse_points=128):
        super(Decoder, self).__init__()
        self.num_coarse = coarse_points
        self.grid_size = 4
        self.num_fine = self.grid_size ** 2 * self.num_coarse

        self.coarse_mlp = MLP([1024, 1024, coarse_points * 3])
        self.final_mlp = MLP_conv([512,512,3])

    def call(self, x):
        coarse = self.coarse_mlp(x)
        coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])

        grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size))
        grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
        grid_feat = tf.tile(grid, [x.shape[0], self.num_coarse, 1])

        point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])

        global_feat = tf.tile(x, [1, self.num_fine, 1])
        feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

        center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = tf.reshape(center, [-1, self.num_fine, 3])

        fine = self.final_mlp(feat) + center

        return coarse, fine