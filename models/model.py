import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from models.encoder import Encoder
from models.decoder import Decoder
from models.model_utils import chamfer_distance


class PCNAutoEncoder(models.Model):
    def __init__(self, latent_size, alpha, debug=False):
        super(PCNAutoEncoder, self).__init__()
        self.alpha = tf.Variable(alpha, trainable=False, dtype=tf.float32)  
        self.encoder = Encoder(latent_size = latent_size, debug=debug)
        self.decoder = Decoder()

    def call(self, inputs):
        x = self.encoder(inputs)
        # x = tf.math.floor(x + 0.5) # Quantize
        x = self.decoder(x)
        return x

    def train_step(self, data):
        inputs = data  

        with tf.GradientTape() as tape:
            coarse_recon, fine_recon = self(inputs, training=True)  
            coarse_loss = tf.reduce_mean(chamfer_distance(inputs, coarse_recon))
            fine_loss = tf.reduce_mean(chamfer_distance(inputs, fine_recon))
            loss_value = coarse_loss + self.alpha * fine_loss

        # Calculate gradients and update model weights
        gradients = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Return a dict mapping metric names to current value for logging purposes
        return {'loss': loss_value, 'recon_loss': fine_loss}
        
    def test_step(self, data):
        inputs = data  
        coarse_recon, fine_recon = self(inputs, training=True)  
        coarse_loss = tf.reduce_mean(chamfer_distance(inputs, coarse_recon))
        fine_loss = tf.reduce_mean(chamfer_distance(inputs, fine_recon))
        loss_value = coarse_loss + self.alpha * fine_loss
        return {'loss': loss_value, 'recon_loss': fine_loss}



