import tensorflow as tf
from tensorflow.keras import layers

from models.model_utils import knn, index_points

class GraphLayer(layers.Layer):
    def __init__(self, out_channel, linear = 64, k=16, relu_out = True, debug=False):
        super(GraphLayer, self).__init__()
        self.k = k
        self.relu_out = relu_out
        self.debug = debug
        self.conv = layers.Conv1D(out_channel, kernel_size=1)
        self.linear = layers.Dense(linear)
        # self.bn = layers.BatchNormalization()

    def call(self, x):
        knn_idx = knn(x, k=self.k)  
        knn_x = index_points(x, knn_idx) 
        if self.debug:
            print(f"graph knn_x: {knn_x.shape}")

        x = tf.reduce_max(knn_x, axis=2)
        if self.debug:
            print(f"global max x: {x.shape}")

        x = self.linear(x)
        if self.debug:
            print(f"global linear x: {x.shape}")

        # Feature Map
        x = self.conv(x)
        if self.debug:
            print(f"global conv x: {x.shape}")
        if self.relu_out:
            x = tf.nn.relu(x)
        return x


class Encoder(layers.Layer):
    def __init__(self, latent_size = 512, k=16, debug=False):
        super(Encoder, self).__init__()
        self.k = k
        self.debug = debug

        self.conv1 = layers.Conv1D(12, kernel_size=1)
        self.conv2 = layers.Conv1D(64, kernel_size=1)
        self.conv3 = layers.Conv1D(64, kernel_size=1)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()

        self.graph_layer1 = GraphLayer(out_channel=128, k=self.k, debug=self.debug)
        self.graph_layer2 = GraphLayer(out_channel=1024, k=self.k, linear=128, relu_out = False, debug=self.debug)

        self.conv4 = layers.Conv1D(1024, kernel_size=1)
        self.conv5 = layers.Conv1D(latent_size, kernel_size=1)

    def call(self, x):
        if self.debug:
            print(f"x: {x.shape}")
        knn_idx = knn(x, k=self.k)  
        knn_x = index_points(x, knn_idx) 

        mean = tf.reduce_mean(knn_x, axis=2, keepdims=True)
        knn_x = knn_x - mean
        if self.debug:
            print(f"knn_x: {knn_x.shape}")


        knn_x_transposed = tf.transpose(knn_x, perm=[0, 1, 3, 2])  # [B, N, C, k]
        covariances = tf.matmul(knn_x_transposed, knn_x)           # [B, N, C, C]
        if self.debug:        
            print(f"covariances: {covariances.shape}")

        # Reshape and concatenate with x
        covariances_flattened = tf.reshape(covariances, [tf.shape(covariances)[0], tf.shape(covariances)[1], tf.shape(covariances)[2]*tf.shape(covariances)[2]])                            # [B, N, C*C]
        if self.debug:        
            print(f"covariances_flattened: {covariances_flattened.shape}")

        x = tf.concat([x, covariances_flattened], axis=2)         # [B, C+C*C, N]
        if self.debug:  
            print(f"x concat: {x.shape}")

        # three layer MLP
        x = tf.nn.relu(self.bn1(self.conv1(x)))
        x = tf.nn.relu(self.bn2(self.conv2(x)))
        x = tf.nn.relu(self.bn3(self.conv3(x)))
        if self.debug:  
            print(f"MLP:{x.shape}")

        # two consecutive graph layers
        x = self.graph_layer1(x)
        x = self.graph_layer2(x)
        if self.debug:  
            print(f"Graph:{x.shape}")

        # Global max pooling
        x = tf.reduce_max(x, axis=1, keepdims=True)
        if self.debug:  
            print(f"Global Max Pool:{x.shape}")

        x = tf.nn.relu(self.conv4(x))
        x = self.conv5(x)
        if self.debug:  
            print(f"Last MLP:{x.shape}")

        return x