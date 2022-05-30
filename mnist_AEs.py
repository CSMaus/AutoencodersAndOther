# doing this as shown in example (AEs paper in Habr, part 1)

from keras.datasets import mnist
import numpy as np

from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model

import seaborn as sns
import matplotlib.pyplot as plt

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


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
    flat_decoded = Dense(28*28, activation='sigmoid')(input_encoded)
    decoder = Reshape((28, 28, 1))(flat_decoded)

    # Models
    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_encoded, decoder, name='decoder')
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
    decoder = Reshape((28, 28, 1))(flat_decoded)

    # Models
    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_encoded, decoder, name='decoder')
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='autoencoder')

    return encoder, decoder, autoencoder


# simple convolutional autoencoder


sc_encoder, sc_decoder, sc_autoencoder = create_deeper_dense_ae()

# Compilation in this case is the construction of a backpropagation calculation graph
sc_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
sc_autoencoder.summary()

sc_autoencoder.fit(x_train, x_train, epochs=10, batch_size=128, shuffle=True, validation_data=(x_test, x_test))


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


n = 10
imgs = x_test[:n]
encoded_imgs = sc_encoder.predict(imgs, batch_size=n)
# encoded_imgs[0]

decoded_imgs = sc_decoder.predict(encoded_imgs, batch_size=n)

plot_digits(imgs, decoded_imgs)

