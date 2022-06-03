import keras.optimizers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from keras.datasets import mnist
from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, Reshape, Lambda
from keras.models import Model


from keras.metrics import binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as bk

# change (random) normal distribution to other random distribution
Z = np.random.randn(150, 2)
X = Z / (np.sqrt(np.sum(Z * Z, axis=1))[:, None]) + Z / 10

fig, axs = plt.subplots(1, 2, sharex=False, figsize=(16, 8))

ax = axs[0]
ax.scatter(Z[:, 0], Z[:, 1])
ax.grid(True)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

ax = axs[1]
ax.scatter(X[:, 0], X[:, 1])
ax.grid(True)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
# plt.show()

# VAE itself

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

batch_size = 256
latent_dim = 2  # to be easier generate and visualize result
dropout_r = 0.2
lr_0 = 0.0001

Adam = keras.optimizers.adam_v2
RMSprop = keras.optimizers.rmsprop_v2


def create_vae(lambda_l1=0.):
    models = {}

    def bn_n_drop(lr):
        return Dropout(dropout_r)(BatchNormalization()(lr))

    # Encoder
    input_img = Input(batch_shape=(batch_size, 28, 28, 1))
    x = Flatten(input_img)
    x = Dense(256, activation='relu')(x)
    x = bn_n_drop(x)
    x = Dense(128, activation='relu')
    x = bn_n_drop(x)

    # predict logarithm of variation instead of standard deviation
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = bk.random_normal(shape=(batch_size, latent_dim), mean=0, stddev=1.0)
        return z_mean + bk.exp(z_log_var / 2) * epsilon

    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    models['encoder'] = Model(input_img, l, 'encoder')
    models['z_meaner'] = Model(input_img, z_mean, 'enc_z_mean')
    models['z_lvarer'] = Model(input_img, z_log_var, 'enc_z_log_var')

    # Decoder
    z = Input(shape=(latent_dim, ))
    x = Dense(128)(z)
    x = LeakyReLU()(x)
    x = bn_n_drop(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = bn_n_drop(x)
    decoded = Dense(28 * 28, activation='sigmoid')(x)

    models['decoder'] = Model(z, decoded, name='decoder')
    models['vae'] = Model(input_img, models['decoder'](models['encoder'](input_img)), name='VAE')

    def vae_loss(x, decoded):
        x = bk.reshape(x, shape=(batch_size, 28 * 28))
        decoded = bk.reshape(decoded, shape=(batch_size, 28*28))
        xent_loss = 28*28*binary_crossentropy(x, decoded)
        kl_loss = -0.5 * bk.sum(1 + z_log_var - bk.square(z_mean) - bk.exp(z_log_var), axis=-1)
        return (xent_loss + kl_loss)/2/28/28

    return models, vae_loss


vae_models, vae_losses = create_vae()
vae = vae_models["vae"]

vae.compile(optimizer='adam', loss=vae_losses)

#Plot images / digits
digit_size = 28

def plot_digits(*args, invert_colors=False):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    figure = np.zeros((digit_size * len(args), digit_size * n))

    for i in range(n):
        for j in range(len(args)):
            figure[j * digit_size: (j + 1) * digit_size,
                   i * digit_size: (i + 1) * digit_size] = args[j][i].squeeze()

    if invert_colors:
        figure = 1-figure

    plt.figure(figsize=(2*n, 2*len(args)))
    plt.imshow(figure, cmap='Greys_r')
    plt.grid(False)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


n = 15  # Img with 15x15 digits
digit_size = 28

from scipy.stats import norm
# Since we are sampling from N(0, I),
# we take the grid of nodes in which we generate numbers from the inverse distribution function
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))


def draw_manifold(generator, show=True):
    figure = np.zeros((digit_size * n, digit_size * n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.zeros((1, latent_dim))
            z_sample[:, :2] = np.array([[xi, yi]])

            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].squeeze()
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    if show:
        # Visualization
        plt.figure(figsize=(15, 15))
        plt.imshow(figure, cmap='Greys_r')
        plt.grid(None)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()
    return figure


from IPython.display import clear_output
from keras.callbacks import LambdaCallback, ReduceLROnPlateau, TensorBoard

# Arrays in which we will save the results for subsequent visualization
figs = []
latent_distrs = []
epochs = []

# Saves epoches
save_epochs = set(list((np.arange(0, 59) ** 1.701).astype(np.int)) + list(range(10)))

# We'll be tracking on these numbers
imgs = x_test[:batch_size]
n_compare = 10

# Models
generator = vae_models["decoder"]
encoder_mean = vae_models["z_meaner"]


# The function that we will run after each epoch
def on_epoch_end(epoch, logs):
    if epoch in save_epochs:
        clear_output()

        # Comparison of real and decoded numbers
        decoded = vae.predict(imgs, batch_size=batch_size)
        plot_digits(imgs[:n_compare], decoded[:n_compare])

        # Manifold drawing
        figure = draw_manifold(generator, show=True)

        # Save variety and z distribution to create animation after
        epochs.append(epoch)
        figs.append(figure)
        latent_distrs.append(encoder_mean.predict(x_test, batch_size))


# Callback
pltfig = LambdaCallback(on_epoch_end=on_epoch_end)
# lr_red = ReduceLROnPlateau(factor=0.1, patience=25)
tb = TensorBoard(log_dir='./logs')

# Run training
vae.fit(x_train, x_train, shuffle=True, epochs=1000,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[pltfig, tb],
        verbose=1)




