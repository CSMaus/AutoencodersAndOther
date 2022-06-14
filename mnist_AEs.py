# doing this as shown in example (AEs paper in Habr, part 1)
# here examples use only MNIST data!
import sys

from keras.datasets import mnist
import numpy as np

from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

import keras.backend as kb
from keras.layers import Lambda

from keras.regularizers import L1L2

import seaborn as sns
import matplotlib.pyplot as plt

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


# draw digits function
def plot_digits(*args):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])

    plt.figure(figsize=(2 * n, 2 * len(args)))
    for j in range(n):
        for i in range(len(args)):
            ax = plt.subplot(len(args), n, i * n + j + 1)
            plt.imshow(args[i][j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


# simplest compressing autoencoder
def create_dense_ae():
    # dimension of the coded representation
    encoding_dim = 49

    # Encoder: input placeholder
    input_img = Input(shape=(28, 28, 1))
    flat_img = Flatten()(input_img)
    encoded = Dense(encoding_dim, activation='relu')(flat_img)

    # Decoder:
    input_encoded = Input(shape=(encoding_dim,))
    flat_decoded = Dense(28 * 28, activation='sigmoid')(input_encoded)
    decoded = Reshape((28, 28, 1))(flat_decoded)

    # Models
    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_encoded, decoded, name='decoder')
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='autoencoder')

    return encoder, decoder, autoencoder


# deeper still dense autoencoder
def create_deeper_dense_ae():
    # dimension of the coded representation
    encoding_dim = 49

    # Encoder: input placeholder
    input_img = Input(shape=(28, 28, 1))
    flat_img = Flatten()(input_img)
    x = Dense(encoding_dim * 3, activation='relu')(flat_img)
    x = Dense(encoding_dim * 2, activation='relu')(x)
    encoded = Dense(encoding_dim, activation='relu')(x)  # activation='linear'

    # Decoder:
    input_encoded = Input(shape=(encoding_dim,))
    x = Dense(encoding_dim * 2, activation='relu')(input_encoded)
    x = Dense(encoding_dim * 3, activation='relu')(x)
    flat_decoded = Dense(28 * 28, activation='sigmoid')(x)
    decoded = Reshape((28, 28, 1))(flat_decoded)

    # Models
    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_encoded, decoded, name='decoder')
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='autoencoder')

    return encoder, decoder, autoencoder


# simple convolutional autoencoder
def create_conv_ae():
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(128, (7, 7), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (7, 7), activation='relu', padding='same')(x)

    input_encoded = Input(shape=(7, 7, 1))
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(input_encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (7, 7), activation='sigmoid', padding='same')(x)

    # Models
    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_encoded, decoded, name='decoder')
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='autoencoder')

    return encoder, decoder, autoencoder


def create_sparse_ae():
    encoding_dim = 16
    lambda_l1 = 0.00001

    # Encoder
    input_img = Input(shape=(28, 28, 1))
    flat_img = Flatten()(input_img)
    x = Dense(encoding_dim * 3, activation='relu')(flat_img)
    x = Dense(encoding_dim * 2, activation='relu')(x)
    encoded = Dense(encoding_dim, activation='linear', activity_regularizer=L1L2(lambda_l1))(x)

    # Decoder:
    input_encoded = Input(shape=(encoding_dim,))
    x = Dense(encoding_dim * 2, activation='relu')(input_encoded)
    x = Dense(encoding_dim * 3, activation='relu')(x)
    flat_decoded = Dense(28 * 28, activation='sigmoid')(x)
    decoded = Reshape((28, 28, 1))(flat_decoded)

    # Models
    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_encoded, decoded, name='decoder')
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='autoencoder')

    return encoder, decoder, autoencoder


def add_noise(x):
    noise_factor = 0.55
    x = x + kb.random_normal(x.get_shape(), 0.5, noise_factor)
    x = kb.clip(x, 0., 1.)
    return x


batch_size = 16


def create_denoising_model(autoencoder):
    input_img = Input(batch_shape=(batch_size, 28, 28, 1))
    noised_img = Lambda(add_noise)(input_img)

    noiser = Model(input_img, noised_img, name='noiser')
    denoiser_model = Model(input_img, autoencoder(noiser(input_img)), name='denoiser')
    return noiser, denoiser_model


s_encoder, s_decoder, s_autoencoder = create_deeper_dense_ae()
s_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

s_autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

n = 5
imgs = x_test[:n]
encoded_imgs = s_encoder.predict(imgs, batch_size=16)
codes = np.vstack([encoded_imgs.mean(axis=0)]*10)
np.fill_diagonal(codes, encoded_imgs.max(axis=0))

decoded_features = s_decoder.predict(codes, batch_size=16)
plot_digits(imgs, decoded_features)

sys.exit()


c_encoder, c_decoder, c_autoencoder = create_deeper_dense_ae()

# Compilation in this case is the construction of a backpropagation calculation graph
c_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
c_autoencoder.summary()

# sc_autoencoder.fit(x_train, x_train, epochs=4, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# get data, add a lot of noise (noise factor ~0.5) and train to restore data from noisy images
c_noiser, c_denoiser_model = create_denoising_model(c_autoencoder)
c_denoiser_model.compile(optimizer='adam', loss='binary_crossentropy')
c_denoiser_model.fit(x_train, x_train,
                     epochs=40,
                     batch_size=batch_size,
                     shuffle=True,
                     validation_data=(x_test, x_test))

n = 10

imgs = x_test[:batch_size]
noised_imgs = c_noiser.predict(imgs, batch_size=batch_size)
encoded_imgs = c_encoder.predict(noised_imgs[:n],  batch_size=n)
decoded_imgs = c_decoder.predict(encoded_imgs[:n], batch_size=n)

plot_digits(imgs[:n], noised_imgs, decoded_imgs)
