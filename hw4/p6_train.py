import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import CSVLogger

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def construct_encoder(latent_dim, num_features):
    encoder_inputs = keras.Input(shape=(num_features, 1))
    x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv1D(16, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

def construct_decoder(latent_dim, num_features):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((16, 64))(x)
    x = layers.Conv1DTranspose(64, 3, activation="relu", strides=2, padding='same')(x)
    x = layers.Conv1DTranspose(32, 3, activation="relu", strides=2, padding='same')(x)
    x = layers.Conv1DTranspose(1, 3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(num_features)(x)
    decoder_outputs = layers.Reshape((num_features, 1))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

def construct_vae(latent_dim, num_features, weights=None):
    encoder = construct_encoder(latent_dim=latent_dim, num_features=num_features)
    decoder = construct_decoder(latent_dim=latent_dim, num_features=num_features)
    encoder.summary()
    decoder.summary()
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    if weights:
        vae.load_weights(weights)
    return vae

def main():
    model_name = 'p6'
    csv_logger = CSVLogger('logs/'+model_name+'.training.log')
    nt_train = np.load('data/nt_train.npy')
    print(nt_train.shape)
    vae = construct_vae(latent_dim=2, num_features=nt_train.shape[1])
    vae.fit(nt_train, epochs=30, batch_size=20, callbacks=[csv_logger])

    vae.save_weights('models/' + model_name + '.model.h5')
    print('Saved model.')

if __name__ == '__main__':
    main()
