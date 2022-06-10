# NOW IT WORKS ONLY WITH KERAS V 2.4.0
import pathlib

import keras.optimizers
import numpy as np, os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from keras.datasets import mnist
from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, Reshape, Lambda
from keras.models import Model

from keras.metrics.metrics import binary_crossentropy
# from keras.objectives import binary_crossentropy
from keras.layers import LeakyReLU
from keras import backend as bk
import tensorflow as tf

import parameters as p
from keras.layers import Rescaling, Reshape, Resizing, RandomZoom, RandomRotation, RandomFlip

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

# PARAMETERS
batch_size = 500
latent_dim = 64  # to be easier generate and visualize result
dropout_r = 0.3
lr_0 = 0.0001
epoch = 50

img_height = p.img_size
img_width = p.img_size

name = f'mnist_dim{latent_dim}_epochs{epoch}'
Adam = keras.optimizers.Adam

# VAE itself

# my data
tf.debugging.set_log_device_placement(True)
list_gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in list_gpu:
    tf.config.experimental.set_memory_growth(gpu, True)

if not p.mnist_d:
    ims = p.img_size
else:
    ims = 28

data_dir = 'D:/DataSets/class2/'
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print('Number of images:', image_count)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

'''train_ds = image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=data_dir,
    shuffle=True,
    target_size=(ims, ims),
    subset="training")

valid_ds = image_generator.flow_from_directory(
    batch_size=32,
    directory=batch_size,
    shuffle=True,
    target_size=(ims, ims),
    subset="validation")'''

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    batch_size=batch_size,
    image_size=(img_height, img_width))

valid_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    batch_size=batch_size,
    image_size=(img_height, img_width))

# train_ds = tf.convert_to_tensor(train_ds)
# sys.exit()
class_names = train_ds.class_names
num_classes = len(class_names)
print(num_classes, class_names)

# Check images and labels
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(5):
    for i in range(6):
        ax = plt.subplot(4, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

data_augmentation = tf.keras.models.Sequential([
    RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    RandomRotation(0.1),
    RandomZoom(0.25)])

# normalization_layer = Rescaling(1. / 255)
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))


# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

train_ds = tf.data.Dataset.from_tensor_slices(list(train_ds))

print('type, shape', np.shape(train_ds))
# sys.exit()

# Image normalization
# normalization_layer = Rescaling(1. / 255)
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]

# train_ds = tf.data.Dataset.from_tensor_slices(normalized_ds).shuffle(1000).batch(batch_size, drop_remainder=True)
# valid_ds = tf.data.Dataset.from_tensor_slices(valid_ds).shuffle(1000).batch(batch_size, drop_remainder=True)
#
# train_ds = np.reshape(train_ds, (len(train_ds), ims, ims, 3))
# valid_ds = np.reshape(valid_ds, (len(valid_ds), ims, ims, 3))


# #########################____MNIST DATA____##################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


# train_ds = tf.data.Dataset.unbatch(train_ds)
# train_ds = Reshape((len(train_ds), ims, ims, 3))

def fix_data(x):
    # x = x.astype(('float32', 'float32'))
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.cache()
    ds = ds.shuffle(1000, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    return ds


# train_ds = fix_data(train_ds)
# print(type(x_train), np.shape(x_train))
# print(type(train_ds), np.shape(train_ds))
# sys.exit()


def create_vae():
    models = {}

    def apply_bn_and_dropout(x):
        return Dropout(dropout_r)(BatchNormalization()(x))

    # Encoder
    input_img = Input(shape=(ims, ims, 3))  # batch_shape=(batch_size, ims, ims, 3)
    # if not p.mnist_d:
    #     # x = data_augmentation(input_img)
    #     x = Rescaling(1. / 255)(input_img)
    #     x = Flatten()(x)
    # else:
    x = Flatten()(input_img)
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


from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

vae_models, vae_losses = create_vae()
vae = vae_models["vae"]

vae.compile(optimizer='adam', loss=vae_losses, experimental_run_tf_function=False)  # vae_losses

# Plot images / digits
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

# from tensorflow.keras.callbacks import LambdaCallback
# Arrays in which we will save the results for subsequent visualization
figs = []
latent_distrs = []
epochs = []

# Saves epoches
save_epochs = set(list((np.arange(0, 59) ** 1.701).astype(int)) + list(range(10)))
# print(save_epochs)
# print(type(save_epochs))
# sys.exit()
# We'll be tracking on these numbers
imgs = train_ds.unbatch().take(batch_size)  # [:batch_size]
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
        latent_distrs.append(encoder_mean.predict(valid_ds, batch_size))


# Callback

lambda_pltfig = LambdaCallback(on_epoch_end=on_epoch_end)

# lr_red = ReduceLROnPlateau(factor=0.1, patience=25)
tb = TensorBoard(log_dir=f'logs/{name}')

# Run training
vae.fit(train_ds, train_ds, shuffle=True, epochs=epoch,
        batch_size=batch_size,
        validation_data=(valid_ds, valid_ds),
        callbacks=[tb],
        verbose=1)

# Comparison of real and decoded numbers
decoded = vae.predict(imgs, batch_size=batch_size)
plot_digits(imgs[:n_compare], decoded[:n_compare])

# Manifold drawing
figure = draw_manifold(generator, show=True)
