import numpy as np, os, sys, matplotlib.pyplot as plt, pathlib

from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, Lambda, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

from keras.metrics.metrics import binary_crossentropy
from keras.layers import LeakyReLU
from keras import backend as bk
import tensorflow as tf

import parameters as p
from keras.layers import Reshape, RandomZoom, RandomRotation, RandomFlip

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# PARAMETERS
batch_size = p.batch_size
latent_dim = p.latent_dim  # make larger to be easier generate and visualize result
dropout_r = p.dropout_r
lr_0 = p.lr_0
epoch = p.epoch

img_height = p.img_size
img_width = p.img_size

tf.debugging.set_log_device_placement(True)
list_gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in list_gpu:
    tf.config.experimental.set_memory_growth(gpu, True)


ims = p.img_size
name = f'pets_dim{latent_dim}_epochs{epoch}_ims_{ims}'


data_dir = p.dataset_folder
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print('Number of images:', image_count)

if batch_size < image_count:
    batch_size = int(image_count * 0.05)
    print(batch_size)
else:
    batch_size = image_count // 2


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
image_generator2 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255, validation_split=0.05)

train_ds = image_generator.flow_from_directory(
    os.path.join(p.dataset_folder),
    class_mode='input',
    target_size=(ims, ims),
    batch_size=batch_size,
    subset="training",
    color_mode='rgb'
)

valid_ds = image_generator.flow_from_directory(
    os.path.join(p.dataset_folder),
    class_mode='input',
    target_size=(ims, ims),
    batch_size=batch_size,
    subset="validation",
    color_mode='rgb'
)

data_augmentation = tf.keras.models.Sequential([
    RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    RandomRotation(0.1),
    RandomZoom(0.25)])


def create_dense_vae():
    models = {}

    def apply_bn_and_dropout(x):
        return Dropout(dropout_r)(BatchNormalization()(x))

    # Encoder
    input_img = Input(shape=(ims, ims, 3))  # batch_shape=(batch_size, ims, ims, 3)
    x = Flatten()(input_img)
    x = Dense(512, activation='relu')(x)
    x = apply_bn_and_dropout(x)
    x = Dense(256, activation='relu')(x)
    x = apply_bn_and_dropout(x)
    x = Dense(128, activation='relu')(x)
    x = apply_bn_and_dropout(x)

    # predict logarithm of variation instead of standard deviation
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    # sampling from Q with reparametrisation
    def sampling(args):
        z_meansа, z_log_varsа = args
        epsilon = bk.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)  #
        return z_meansа + bk.exp(z_log_varsа) * epsilon

    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = Model(input_img, l, name='my_encoder')
    models["encoder"] = encoder
    models["z_meaner"] = Model(input_img, z_mean, 'Enc_z_mean')
    models["z_lvarer"] = Model(input_img, z_log_var, 'Enc_z_log_var')

    # Decoder
    z = Input(shape=(latent_dim,))
    x = Dense(128)(z)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(512)(x)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(ims * ims * 3, activation='sigmoid')(x)
    decoded = Reshape((ims, ims, 3))(x)

    decoder = Model(z, decoded, name='my_decoder')

    models["decoder"] = decoder
    outputs = decoder(encoder(input_img))

    my_vae = Model(inputs=input_img, outputs=outputs, name='my_vae')
    models["vae"] = my_vae

    def vae_loss(x1, decoded1):
        x1 = bk.reshape(x1, shape=(batch_size, ims * ims * 3))
        decoded1 = bk.reshape(decoded1, shape=(batch_size, ims * ims * 3))
        xent_loss = ims * ims * 3 * binary_crossentropy(x1, decoded1)
        bkl_loss = -0.5 * bk.sum(1 + z_log_var - bk.square(z_mean) - bk.exp(z_log_var), axis=-1)
        return (xent_loss + bkl_loss) / 2 / ims / ims / 3

    return models, vae_loss


def create_simple_conv_vae():
    models = {}

    def apply_bn_and_dropout(x):
        return Dropout(dropout_r)(BatchNormalization()(x))

    # Encoder
    input_img = Input(shape=(ims, ims, 3))  # batch_shape=(batch_size, ims, ims, 3)
    # x = Flatten()(input_img)
    x = Conv2D(128, (7, 7), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1, (7, 7), activation='relu', padding='same')(x)

    # predict logarithm of variation instead of standard deviation
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    # sampling from Q with reparametrisation
    def sampling(args):
        z_meansа, z_log_varsа = args
        epsilon = bk.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)  #
        return z_meansа + bk.exp(z_log_varsа) * epsilon

    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = Model(input_img, l, name='my_encoder')
    models["encoder"] = encoder
    models["z_meaner"] = Model(input_img, z_mean, 'Enc_z_mean')
    models["z_lvarer"] = Model(input_img, z_log_var, 'Enc_z_log_var')

    # Decoder
    z = Input(shape=(7, 7, 1))
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(z)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (7, 7), activation='sigmoid', padding='same')(x)

    decoder = Model(z, decoded, name='my_decoder')

    models["decoder"] = decoder
    outputs = decoder(encoder(input_img))

    my_vae = Model(inputs=input_img, outputs=outputs, name='my_vae')
    models["vae"] = my_vae

    def vae_loss(x1, decoded1):
        x1 = bk.reshape(x1, shape=(batch_size, ims * ims * 3))
        decoded1 = bk.reshape(decoded1, shape=(batch_size, ims * ims * 3))
        xent_loss = ims * ims * 3 * binary_crossentropy(x1, decoded1)
        bkl_loss = -0.5 * bk.sum(1 + z_log_var - bk.square(z_mean) - bk.exp(z_log_var), axis=-1)
        return (xent_loss + bkl_loss) / 2 / ims / ims / 3

    return models, vae_loss


datadir = 'D:/DataSets/dogs_cats/'
train_ds2 = tf.keras.utils.image_dataset_from_directory(
    datadir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    datadir,
    validation_split=0.05,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds2.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds2.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# plt.show()
print(type(images))
print(images.shape)
print(type(images.numpy()))
images = images.numpy().astype("float32")

images = images/255.
# images = images.astype("uint8")

# ######################################################################################################
# ############################## CREATE, COMPILE AND FIT VAE ###########################################

from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
disable_eager_execution()
vae_models, vae_losses = create_dense_vae()
vae = vae_models["vae"]


# vae.compile(optimizer='adam', loss=vae_losses, experimental_run_tf_function=False)  # vae_losses
vae.compile(optimizer='adam', loss=vae_losses)
# Plot images / digits
if p.mnist_d:
    digit_size = 28
else:
    digit_size = ims


def plot_digits(*args, invert_colors=False):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    figure = np.zeros((digit_size * len(args), digit_size * n, 3))

    for i in range(n):
        for j in range(len(args)):
            figure[j * digit_size: (j + 1) * digit_size,
            i * digit_size: (i + 1) * digit_size, :] = args[j][i].squeeze()

    if invert_colors:
        figure = 1 - figure

    plt.figure(figsize=(2 * n, 2 * len(args)))
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


def draw_manifold(generator, show=True):
    figure = np.zeros((digit_size * n, digit_size * n, 3))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.zeros((1, latent_dim))
            z_sample[:, :2] = np.array([[xi, yi]])

            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].squeeze()
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size, :] = digit
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


from keras.callbacks import LambdaCallback, ReduceLROnPlateau, TensorBoard

# from tensorflow.keras.callbacks import LambdaCallback
# Arrays in which we will save the results for subsequent visualization
figs = []
latent_distrs = []
epochs = []

# Saves epoches
save_epochs = set(list((np.arange(0, 59) ** 1.701).astype(int)) + list(range(10)))

# We'll be tracking on these numbers
n_compare = 5

# Models
generator = vae_models["decoder"]
encoder_mean = vae_models["z_meaner"]

tb = TensorBoard(log_dir=f'logs/{name}')

# Run training
vae.fit(train_ds, shuffle=True, epochs=epoch,
        validation_data=valid_ds,
        callbacks=[tb],
        verbose=1)


# enable_eager_execution()
decoded = vae.predict(images, batch_size=batch_size, steps=1)  # , steps=1
# plot_digits(image, decoded)
plot_digits(images[:n_compare], decoded[:n_compare])
# Manifold drawing
figure = draw_manifold(generator, show=True)

# Comparison of real and decoded numbers
# decoded = vae.predict(imgs, batch_size=batch_size)
# plot_digits(imgs[:n_compare], decoded[:n_compare])
# decoded = vae.predict(valid_ds)
# plot_digits(valid_ds, decoded)
#
# # Manifold drawing
# figure = draw_manifold(generator, show=True)
