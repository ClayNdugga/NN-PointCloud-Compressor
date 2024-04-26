import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers

# from models.FNencoder import Encoder
from models.PCNEncoder import Encoder
from models.decoder import Decoder
from models.model_utils import chamfer_distance


class PCNAutoEncoder(models.Model):
    def __init__(self, latent_size, alpha, lmbda=0, N=35):  
        super(PCNAutoEncoder, self).__init__()
        self.alpha = tf.Variable(alpha, trainable=False, name="alpha", dtype=tf.float32)  
        self.lmbda = float(lmbda)
        self.N = N

        self.mu = tf.Variable(tf.zeros([1, 1024]), trainable=False, name="mean", dtype=tf.float32)
        self.variance = tf.Variable(tf.ones([1, 1024]), trainable=False, name="variance", dtype=tf.float32)  

        self.encoder = Encoder()
        self.decoder = Decoder()
        

    def call(self, inputs, training=False):
        x = self.encoder(inputs)
        if training:
            x_quantized = self.soft_quantizer(x)
        else:
            x_quantized = self.hard_quantizer(x)
        x = self.decoder(x_quantized)
        return x, x_quantized
    
    
    
    def soft_quantizer(self, x, bound=4.5):
        quantization_level_size = (2*bound) / self.N
        noise = tf.random.uniform(tf.shape(x), minval=-quantization_level_size/2, maxval=quantization_level_size/2)
        x_noisy = x + noise
        return x_noisy
    
    def hard_quantizer(self, x, bounds=4.5):
        quantized_levels = tf.linspace(float(-bounds), float(bounds), self.N)
        x_expanded = tf.expand_dims(x, -1)
        quantized_levels_expanded = tf.reshape(quantized_levels, (1, 1, -1))
        indices = tf.argmin(tf.abs(x_expanded - quantized_levels_expanded), axis = -1)
        data = tf.gather(quantized_levels, indices)
        return data
    
    def lossless_coder(self, data, mu=0.0, sigma=1.0):
        pi = tf.constant(np.pi, dtype=tf.float32)
        log_prob = 0.5 * tf.math.log(2.0 * pi * (self.variance + float(1e-6))) + (0.5 * (tf.math.log(np.e) * (data - self.mu) ** 2  / (self.variance + float(1e-6))))
        total_log_probs = tf.reduce_sum(log_prob, axis=2)
        batch_rate = tf.reduce_mean(total_log_probs)
        return batch_rate



    def train_step(self, data):
        with tf.GradientTape() as tape:
            (coarse_recon, fine_recon), quatized_data = self(data, training=True)          
            coarse_loss = tf.reduce_mean(chamfer_distance(data, coarse_recon))
            fine_loss = tf.reduce_mean(chamfer_distance(data, fine_recon))
            rate = self.lossless_coder(quatized_data)

            loss_value = coarse_loss + (self.alpha * fine_loss) + (self.lmbda * rate)

        gradients = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': loss_value, 'recon_loss': fine_loss, 'code_rate': rate}
        
    def test_step(self, data): 
        (coarse_recon, fine_recon), quatized_data = self(data, training=False)            
        coarse_loss = tf.reduce_mean(chamfer_distance(data, coarse_recon))
        fine_loss = tf.reduce_mean(chamfer_distance(data, fine_recon))
        
        rate = self.lossless_coder(quatized_data)

        loss_value = coarse_loss + (self.alpha * fine_loss) + (self.lmbda * rate)
        return {'loss': loss_value, 'recon_loss': fine_loss, 'code_rate': rate}

