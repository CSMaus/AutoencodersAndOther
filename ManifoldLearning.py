import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.regularizers import L1L2
from keras.datasets import mnist
from keras.models import Model
from keras.optimizers import adam_v2

from sklearn.decomposition import PCA
from itertools import cycle
import parameters as p


# linear autoencoder: to find linear dependencies in data
def linear_ae():
    input_dots = Input((2,))
    code = Dense(1, activation='linear')(input_dots)
    out = Dense(2, activation='linear')(code)

    model = Model(input_dots, out)
    return model


def deeper_ae():
    input_dots = Input((2,))
    x = Dense(64, activation='elu')(input_dots)
    x = Dense(64, activation='elu')(x)
    code = Dense(1, activation='linear')(x)
    x = Dense(64, activation='elu')(code)
    x = Dense(64, activation='elu')(x)
    out = Dense(2, activation='linear')(x)

    model = Model(input_dots, out)
    return model


def deep_conv_ae():
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

    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_encoded, decoded, name='decoder')
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='autoencoder')

    return encoder, decoder, autoencoder


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


def deep_sparse_ae(lambda_l1=0.):
    encoding_dim = 16

    input_img = Input(shape=(28, 28, 1))
    flat_img = Flatten()(input_img)
    x = Dense(encoding_dim * 4, activation='relu')(flat_img)
    x = Dense(encoding_dim * 3, activation='relu')(x)
    x = Dense(encoding_dim * 2, activation='relu')(x)
    encoded = Dense(encoding_dim, activation='linear', activity_regularizer=L1L2(lambda_l1, 0))(x)

    input_encoded = Input(shape=(encoding_dim,))
    x = Dense(encoding_dim * 2, activation='relu')(input_encoded)
    x = Dense(encoding_dim * 3, activation='relu')(x)
    x = Dense(encoding_dim * 4, activation='relu')(x)
    flat_decoded = Dense(28 * 28, activation='sigmoid')(x)
    decoded = Reshape((28, 28, 1))(flat_decoded)

    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_encoded, decoded, name='decoder')
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='autoencoder')

    return encoder, decoder, autoencoder


# Homotopy straight between objects or between codes
def plot_homotopy(frm, to, n=p.n, decoder=None):
    z = np.zeros(([n] + list(frm.shape)))
    for i, t in enumerate(np.linspace(0., 1., n)):
        z[i] = frm * (1-t) + to * t
    if decoder:
        plot_digits(decoder.predict(z, batch_size=n))
    else:
        plot_digits(z)


if p.using_dot_generator:

    # Creating 2-dimensional dataset as a curve and noise and will use them to train ae
    x1 = np.linspace(-2.2, 2.2, 1000)
    fx = np.sin(x1)
    dots = np.vstack([x1, fx]).T
    noise = 0.06 * np.random.randn(*dots.shape)  # Return a sample (or samples) from the "standard normal" distribution
    dots += noise

    # colored dots for visualisation
    size = 25
    colors = ['r', 'g', 'c', 'y', 'm']
    idxs = range(0, x1.shape[0], x1.shape[0] // size)
    vx1 = x1[idxs]
    vdots = dots[idxs]

    ae = deeper_ae()
    # optimizer = adam_v2.Adam(learning_rate=lr, decay=lr/epochs)
    ae.compile('adam', 'mse')
    ae.fit(dots, dots, epochs=200, batch_size=30)

    # using ae
    pdots = ae.predict(dots, batch_size=30)
    vpdots = pdots[idxs]
    pca = PCA(1)
    pdots_pca = pca.inverse_transform(pca.fit_transform(dots))

    # visualisation
    plt.figure(figsize=(12, 10))
    plt.xlim([-2.5, 2.5])
    plt.scatter(dots[:, 0], dots[:, 1], zorder=1)
    plt.plot(x1, fx, color="red", linewidth=4, zorder=10)
    plt.plot(pdots[:, 0], pdots[:, 1], color='gray', linewidth=12, zorder=3)
    plt.plot(pdots_pca[:, 0], pdots_pca[:, 1], color='orange', linewidth=4, zorder=4)
    plt.scatter(vpdots[:, 0], vpdots[:, 1], color=colors * 5, marker='*', s=150, zorder=5)
    plt.scatter(vdots[:, 0], vdots[:, 1], color=colors * 5, s=150, zorder=6)
    plt.grid(True)
    plt.show()
    '''
    grey line - the manifold into which the blue data points go after the autoencoder,
        that is, the autoencoder's attempt to build a manifold that determines the most variations in the data
    orange line - the manifold into which the blue data points go after PCA
    multi-colored circles - dots that turn into stars of the corresponding color after the autoencoder
    multi-colored stars - respectively, the images of circles after the autoencoder
    
    the closer the gray line is to the red the ae is better.
    '''
else:
    # load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    c_encoder, c_decoder, c_autoencoder = deep_sparse_ae()

    c_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    c_autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
                      validation_data=(x_test, x_test))

    imgs = x_test[:p.n]
    decoded_imgs = c_autoencoder.predict(imgs, batch_size=p.n)

    plot_digits(imgs, decoded_imgs)

    sys.exit()
    # Homotopy between first two 8
    frm, to = x_test[y_test == 8][1:3]
    plot_homotopy(frm, to)

    codes = c_encoder.predict(x_test[y_test == 8][1:3])
    plot_homotopy(codes[0], codes[1], n=p.n, decoder=c_decoder)










