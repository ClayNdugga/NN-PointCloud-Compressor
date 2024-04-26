import tensorflow as tf
from tensorflow.keras import layers

###################################### General Utils ######################################

@tf.keras.utils.register_keras_serializable(package="AutoEncoder", name="MLP")
class MLP(layers.Layer):
    def __init__(self, layer_dims, batch_normalization = False):
        super(MLP, self).__init__()
        self.layer_dims = layer_dims
        self.fully_connected_layers = [layers.Dense(layer_dim) for layer_dim in self.layer_dims[:-1]]
        self.batch_layers = [layers.BatchNormalization() for _ in self.layer_dims[:-1]]
        self.final_dense = layers.Dense(self.layer_dims[-1])
        self.batch_normalization = batch_normalization

    def call(self, x):
        for dense, batch_norm in zip(self.fully_connected_layers, self.batch_layers):
            x = dense(x)
            if self.batch_normalization:
                x = batch_norm(x)
            # x = tf.nn.relu(x)
        x = self.final_dense(x)
        return x
    
    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            'layer_dims': self.layer_dims,
            'batch_normalization': self.batch_normalization,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="AutoEncoder", name="MLP_conv")
class MLP_conv(layers.Layer):
    def __init__(self, layer_dims, batch_normalization = False):
        super(MLP_conv, self).__init__()
        self.layer_dims = layer_dims
        self.conv_layers = [layers.Conv1D(layer_dim, kernel_size = 1) for layer_dim in self.layer_dims[:-1]]
        self.batch_layers = [layers.BatchNormalization() for _ in self.layer_dims[:-1]]
        self.final_conv = layers.Conv1D(self.layer_dims[-1], kernel_size = 1)
        self.batch_normalization = batch_normalization

    def call(self, x):
        for conv, batch_norm in zip(self.conv_layers, self.batch_layers):
            x = conv(x)
            if self.batch_normalization:
                x = batch_norm(x)
            x = tf.nn.relu(x)
        x = self.final_conv(x)
        return x
    
    def get_config(self):
        config = super(MLP_conv, self).get_config()
        config.update({
            'layer_dims': self.layer_dims,
            'batch_normalization': self.batch_normalization,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

###################################### FN Encoder Utils ######################################


def knn(x, k):
    # Compute pairwise distance
    x_transpose = tf.transpose(x, perm=[0, 2, 1])
    product = tf.matmul(x, x_transpose)
    square = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    pairwise_distance = square + tf.transpose(square, perm=[0, 2, 1]) - 2 * product

    _, idx = tf.nn.top_k(-pairwise_distance, k=k, sorted=True)
    return idx

def index_points(point_clouds, indices):
    batch_size = tf.shape(point_clouds)[0]
    num_points = tf.shape(indices)[1]
    k = tf.shape(indices)[2]

    # Create a batch index tensor````
    batch_indices = tf.range(batch_size)
    batch_indices = tf.reshape(batch_indices, [batch_size, 1, 1])
    batch_indices = tf.tile(batch_indices, [1, num_points, k])

    # Combine batch indices with indices
    indices = tf.stack([batch_indices, indices], axis=-1)

    # Use tf.gather_nd to gather points
    new_points = tf.gather_nd(point_clouds, indices)

    return new_points



###################################### Loss Function ######################################


def chamfer_distance(set1, set2):
    # Compute pairwise distances
    s1_expand = tf.expand_dims(set1, axis=2)
    s2_expand = tf.expand_dims(set2, axis=1)
    distances = tf.reduce_sum(tf.square(s1_expand - s2_expand), axis=-1)

    # Compute minimum distances
    min_dist_1_to_2 = tf.reduce_min(distances, axis=2)
    min_dist_2_to_1 = tf.reduce_min(distances, axis=1)

    # Average minimum distances
    chamfer_dist = tf.reduce_mean(min_dist_1_to_2, axis=1) + tf.reduce_mean(min_dist_2_to_1, axis=1)
    return chamfer_dist