import tensorflow as tf
from tensorflow.keras import layers
from model_utils import MLP, MLP_conv


@tf.keras.utils.register_keras_serializable(package="AutoEncoder", name="Encoder2")
class Encoder(layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

        self.mlp1_config = {'layer_dims':[128,256]}
        self.mlp2_config = {'layer_dims':[512,1024]}
        self.mlp1 = MLP_conv(**self.mlp1_config)
        self.mlp2 = MLP_conv(**self.mlp2_config)


    def call(self, x):
        features  = self.mlp1(x)
        features_global = tf.reduce_max(features, axis=1, keepdims=True)
        features = tf.concat([features, tf.tile(features_global, [1, tf.shape(x)[1], 1])], axis=2)

        features = self.mlp2(features)
        features = tf.reduce_max(features, axis=1,keepdims=True)
        
        return features

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'mlp1_config': self.mlp1_config,
            'mlp2_config': self.mlp2_config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['mlp1_config'] = config.get('mlp1_config', {})
        config['mlp2_config'] = config.get('mlp2_config', {})
        
        return cls(**config)
        