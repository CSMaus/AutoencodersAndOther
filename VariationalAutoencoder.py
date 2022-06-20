# NOW IT WORKS ONLY WITH KERAS V 2.4.0
import keras.optimizers
import numpy as np, os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from keras.datasets import mnist
from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, Reshape, Lambda, Concatenate
from keras.models import Model

import tensorflow as tf
from keras.metrics.metrics import binary_crossentropy
from keras.layers import LeakyReLU
from keras import backend as bk
# from tensorflow.keras.utils import to_categorical
import parameters as p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# print(type(x_train))
# print(np.shape(x_train))
# sys.exit()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

y_train_cat = tf.keras.utils.to_categorical(y_train).astype(np.float32)
y_test_cat = tf.keras.utils.to_categorical(y_test).astype(np.float32)
num_classes = y_test_cat.shape[1]

print(np.shape(x_train))
print(type(x_train))

print(np.shape(x_test))
print(type(x_test))

# sys.exit()

batch_size = 500
latent_dim = 64  # to be easier generate and visualize result
dropout_r = 0.3
lr_0 = 0.0001
epoch = 10

name = f'mnist_dim{latent_dim}_epochs{epoch}'

Adam = keras.optimizers.Adam
# Adam = keras.optimizers.adam
# RMSprop = keras.optimizers.rmsprop


# remake to conditional VAE
def create_cvae():
    models = {}

    def apply_bn_and_dropout(x):
        return Dropout(dropout_r)(BatchNormalization()(x))

    # Encoder
    inp_img = Input(shape=(28, 28, 1))  # batch_shape=(batch_size, 28, 28, 1)
    flat = Flatten()(inp_img)
    inp_lbls = Input(shape=(num_classes, ), dtype='float32')

    x = Concatenate()([flat, inp_lbls])
    x = Dense(256, activation='relu')(x)
    x = apply_bn_and_dropout(x)
    x = Dense(128, activation='relu')(x)
    x = apply_bn_and_dropout(x)

    # predict logarithm of variation instead of standard deviation
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    # sampling from Q with reparametrisation
    def sampling(args):
        z_means, z_log_vars = args
        epsilon = bk.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_means + bk.exp(z_log_vars / 2) * epsilon

    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = Model([inp_img, inp_lbls], l, name='my_encoder')
    z_meaner = Model([inp_img, inp_lbls], z_mean, name='Enc_z_mean')
    models["encoder"] = encoder
    models["z_meaner"] = z_meaner
    models["z_lvarer"] = Model([inp_img, inp_lbls], z_log_var, name='Enc_z_log_var')

    # Decoder
    z = Input(shape=(latent_dim,))
    inp_lbls_d = Input(shape=(num_classes, ), dtype='float32')
    x = Concatenate()([z, inp_lbls_d])
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(28 * 28, activation='sigmoid')(x)
    decoded = Reshape((28, 28, 1))(x)

    decoder = Model([z, inp_lbls_d], decoded, name='my_decoder')

    models["decoder"] = decoder

    # for vae
    # outputs = decoder(encoder(input_img))
    # my_vae = Model(inputs=input_img, outputs=outputs, name='my_vae')
    # models["vae"] = my_vae

    cvae_out = decoder([encoder([inp_img, inp_lbls]), inp_lbls_d])
    my_cvae = Model([inp_img, inp_lbls, inp_lbls_d], cvae_out, name='my_cvae')
    models['cvae'] = my_cvae

    out_style = decoder([z_meaner([inp_img, inp_lbls]), inp_lbls_d])
    models["style_t"] = Model([inp_img, inp_lbls, inp_lbls_d], out_style, name="style_transfer")

    def vae_loss(x1, decoded1):
        x1 = bk.reshape(x1, shape=(batch_size, 28 * 28))
        decoded1 = bk.reshape(decoded1, shape=(batch_size, 28 * 28))
        xent_loss = 28 * 28 * binary_crossentropy(x1, decoded1)
        bkl_loss = -0.5 * bk.sum(1 + z_log_var - bk.square(z_mean) - bk.exp(z_log_var), axis=-1)
        return (xent_loss + bkl_loss) / 2 / 28 / 28

    return models, vae_loss


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


cvae_models, cvae_losses = create_cvae()
cvae = cvae_models["cvae"]

cvae.compile(optimizer='adam', loss=cvae_losses, experimental_run_tf_function=False)  # cvae_losses

# Plot images / digits
digit_size = 28


def plot_digits(*args, invert_colors=False):
    args = [x.squeeze() for x in args]
    n_f = min([x.shape[0] for x in args])
    figure = np.zeros((digit_size * len(args), digit_size * n_f))

    for i in range(n_f):
        for j in range(len(args)):
            figure[j * digit_size: (j + 1) * digit_size,
                   i * digit_size: (i + 1) * digit_size] = args[j][i].squeeze()

    if invert_colors:
        figure = 1 - figure

    plt.figure(figsize=(2 * n_f, 2 * len(args)))
    plt.imshow(figure, cmap='Greys_r')
    plt.grid(False)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


n = 15  # Img with 15x15 digits

from scipy.stats import norm

# Since we are sampling from N(0, I),
# we take the grid of nodes in which we generate numbers from the inverse distribution function
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))


if p.vae:
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
else:

    from scipy.stats import norm
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    def draw_manifold(generator, lbl, show=True):
        figure = np.zeros((digit_size * n, digit_size * n))
        input_lbl = np.zeros((1, 10))
        input_lbl[0, lbl] = 1
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.zeros((1, latent_dim))
                z_sample[:, :2] = np.array([[xi, yi]])

                x_decoded = generator.predict([z_sample, input_lbl])
                digit = x_decoded[0].squeeze()
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit
        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(figure, cmap='Greys_r')
            plt.grid(False)
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.show()
        return figure


    def draw_z_distr(z_predicted, lbl):
        input_lbl = np.zeros((1, 10))
        input_lbl[0, lbl] = 1
        im = plt.scatter(z_predicted[:, 0], z_predicted[:, 1])
        im.axes.set_xlim(-5, 5)
        im.axes.set_ylim(-5, 5)
        plt.show()


from IPython.display import clear_output
from keras.callbacks import LambdaCallback, ReduceLROnPlateau, TensorBoard
# from tensorflow.keras.callbacks import LambdaCallback

# Arrays in which we will save the results for subsequent visualization
# figs = []
# latent_distrs = []
figs = [[] for x in range(num_classes)]
latent_distrs = [[] for x in range(num_classes)]
epochs = []

# Saves epoches
save_epochs = set(list((np.arange(0, 59) ** 1.701).astype(int)) + list(range(10)))

# We'll be tracking on these numbers
imgs = x_test[:batch_size]
imgs_lbls = y_test_cat[:batch_size]
n_compare = 10

# Models
generator = cvae_models["decoder"]
encoder_mean = cvae_models["z_meaner"]


# The function that we will run after each epoch
if p.vae:
    def on_epoch_end(epoch, logs):
        if epoch in save_epochs:
            clear_output()

            # Comparison of real and decoded numbers
            decoded = cvae.predict(imgs, batch_size=batch_size)
            plot_digits(imgs[:n_compare], decoded[:n_compare])

            # Manifold drawing
            figure = draw_manifold(generator, show=True)

            # Save variety and z distribution to create animation after
            epochs.append(epoch)
            figs.append(figure)
            latent_distrs.append(encoder_mean.predict(x_test, batch_size))
else:
    def on_epoch_end(epoch, logs):
        if epoch in save_epochs:
            clear_output()  # Не захламляем output

            # Сравнение реальных и декодированных цифр
            decoded = cvae.predict([imgs, imgs_lbls, imgs_lbls], batch_size=batch_size)
            plot_digits(imgs[:n_compare], decoded[:n_compare])

            # Рисование многообразия для рандомного y и распределения z|y
            draw_lbl = np.random.randint(0, num_classes)
            print(draw_lbl)
            for lbl in range(num_classes):
                figs[lbl].append(draw_manifold(generator, lbl, show=lbl == draw_lbl))
                idxs = y_test == lbl
                z_predicted = encoder_mean.predict([x_test[idxs], y_test_cat[idxs]], batch_size)
                latent_distrs[lbl].append(z_predicted)
                if lbl == draw_lbl:
                    draw_z_distr(z_predicted, lbl)
            epochs.append(epoch)

# Callback

lambda_pltfig = LambdaCallback(on_epoch_end=on_epoch_end)

# lr_red = ReduceLROnPlateau(factor=0.1, patience=25)
tb = TensorBoard(log_dir=f'logs/{name}')

# Run training
cvae.fit([x_train, y_train_cat, y_train_cat], x_train, shuffle=True, epochs=epoch,
         batch_size=batch_size,
         validation_data=([x_test, y_test_cat, y_test_cat], x_test),
         callbacks=[tb],
         verbose=1)


def style_transfer(model, X, lbl_in, lbl_out):
    rows = X.shape[0]
    if isinstance(lbl_in, int):
        lbl_f = lbl_in
        lbl_in = np.zeros((rows, 10))
        lbl_in[:, lbl_f] = 1
    if isinstance(lbl_out, int):
        lbl_f = lbl_out
        lbl_out = np.zeros((rows, 10))
        lbl_out[:, lbl_f] = 1
    return model.predict([X, lbl_in, lbl_out])


n = 10
lbl = 7
generated = []
prot = x_train[y_train == lbl][:n]

for i in range(num_classes):
    generated.append(style_transfer(cvae_models["style_t"], prot, lbl, i))

generated[lbl] = prot
plot_digits(*generated, invert_colors=True)

# Comparison of real and decoded numbers
print(type(imgs))
print(imgs.shape)
decoded = cvae.predict(imgs, batch_size=batch_size)
plot_digits(imgs[:n_compare], decoded[:n_compare])

# Manifold drawing
figure = draw_manifold(generator, show=True)


