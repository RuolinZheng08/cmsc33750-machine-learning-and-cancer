import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import CSVLogger
from sklearn.metrics import accuracy_score

from train import construct_model

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
    x = layers.Dense(64, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

def construct_decoder(latent_dim, num_features):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((32, 64))(x)
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

def construct_vae(latent_dim, num_features):
    encoder = construct_encoder(latent_dim=latent_dim, num_features=num_features)
    decoder = construct_decoder(latent_dim=latent_dim, num_features=num_features)
    encoder.summary()
    decoder.summary()
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    return vae

def main():
    model_name = 'p6'
    csv_logger = CSVLogger('logs/'+model_name+'.training.log', append=True)

    # load training data
    nt_train = np.load('data/nt_train.npy')
    print(nt_train.shape)

    # construct testing data
    nt_test = np.load('data/nt_test.npy')
    # use labels to pull out normal vs tumor
    nt_test_labels = np.load('data/nt_test_labels.npy')
    test_normal_indices = (nt_test_labels[:, 1] == 0)
    test_tumor_indices = (nt_test_labels[:, 1] == 1)
    nt_test_normal = nt_test[test_normal_indices]
    nt_test_tumor = nt_test[test_tumor_indices]

    # build model
    vae = construct_vae(latent_dim=2, num_features=nt_train.shape[1])

    # 37 classes: 36 + normal from part 2
    p2_model = construct_model(nt_train.shape[1], 37, weights='models/p2.autosave.model.h5')

    for checkpoint in range(1,11): # train for a total of 300 epochs
        vae.fit(nt_train, epochs=30, batch_size=20, callbacks=[csv_logger])

        vae.encoder.save_weights('models/{}.encoder{}.model.h5'.format(model_name, checkpoint))
        vae.decoder.save_weights('models/{}.decoder{}.model.h5'.format(model_name, checkpoint))

        # evaluate using the classifier from part 2
        # sample from VAE
        vae_sampled_normal = vae.decoder.predict(
            vae.encoder.predict(nt_test_normal)
        )
        vae_sampled_tumor = vae.decoder.predict(
            vae.encoder.predict(nt_test_tumor)
        )
        print(vae_sampled_normal.shape, vae_sampled_tumor.shape)

        X_test = np.concatenate([vae_sampled_normal, vae_sampled_tumor])
        Y_test = np.concatenate([
            np.full(vae_sampled_normal.shape[0], fill_value=0),
            np.full(vae_sampled_tumor.shape[0], fill_value=1)
        ])

        Y_preds = p2_model.predict(X_test, verbose=0)
        Y_preds = np.argmax(Y_preds, axis=1)
        # like in part 3, set tumor samples i.e. any label that
        # is not 36 normal to 1 and normal 36 to 0
        Y_preds[Y_preds != 36] = 1
        Y_preds[Y_preds == 36] = 0

        print('** Accuracy using part 2 model', accuracy_score(Y_test, Y_preds), end='\n\n')

    # load model to make sure the weights are okay
    vae2 = construct_vae(latent_dim=2, num_features=nt_train.shape[1])
    vae2.encoder.load_weights('models/' + model_name + '.encoder5.model.h5')
    vae2.decoder.load_weights('models/' + model_name + '.decoder5.model.h5')

    print('Save okay')

if __name__ == '__main__':
    main()
