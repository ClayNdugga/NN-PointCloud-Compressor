import tensorflow as tf
from tensorflow.keras import layers

from model_utils import MLP, MLP_conv
    
    
@tf.keras.utils.register_keras_serializable(package="AutoEncoder", name="Decoder")
class Decoder(layers.Layer):
    def __init__(self, coarse_points=256):
        super(Decoder, self).__init__()
        self.num_coarse = coarse_points
        self.grid_size = 4
        self.num_fine = self.grid_size ** 2 * self.num_coarse

        self.coarse_mlp_config = {'layer_dims': [1024, 1024, self.num_coarse * 3], 'batch_normalization': False}
        self.final_mlp_config = {'layer_dims': [512, 512, 3], 'batch_normalization': False}
        self.coarse_mlp = MLP(**self.coarse_mlp_config)
        self.final_mlp = MLP_conv(**self.final_mlp_config)

    def call(self, x):
        coarse = self.coarse_mlp(x)
        coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])

        grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size))
        grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
        grid_feat = tf.tile(grid, [tf.shape(x)[0], self.num_coarse, 1])

        point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])

        global_feat = tf.tile(x, [1, self.num_fine, 1])
        feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

        center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = tf.reshape(center, [-1, self.num_fine, 3])

        fine = self.final_mlp(feat) + center

        return coarse, fine
    
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'coarse_points': self.num_coarse,
            'coarse_mlp_config': self.coarse_mlp_config,
            'final_mlp_config': self.final_mlp_config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['coarse_mlp_config'] = config.get('coarse_mlp_config', {})
        config['final_mlp_config'] = config.get('final_mlp_config', {})
        
        return cls(**config)