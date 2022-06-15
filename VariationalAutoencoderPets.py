# NOW IT WORKS ONLY WITH KERAS V 2.4.0
import pathlib
import numpy as np, os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
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

'''
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
'''

# PARAMETERS
batch_size = p.batch_size
latent_dim = p.latent_dim  # to be easier generate and visualize result
dropout_r = p.dropout_r
lr_0 = p.lr_0
epoch = p.epoch

img_height = p.img_size
img_width = p.img_size

# VAE itself

# my data
tf.debugging.set_log_device_placement(True)
list_gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in list_gpu:
    tf.config.experimental.set_memory_growth(gpu, True)

if not p.mnist_d:
    name = f'pets_dim{latent_dim}_epochs{epoch}'
    ims = p.img_size
else:
    name = f'mnist_dim{latent_dim}_epochs{epoch}'
    ims = 28

data_dir = p.dataset_folder
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print('Number of images:', image_count)
# sys.exit()
if batch_size < image_count:
    # batch_size = batch_size
    batch_size = int(image_count * 0.05)
    print(batch_size)
else:
    batch_size = image_count // 2
# sys.exit()

if p.use_flow:
    if p.use_flow_from_directory:
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

    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
else:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        batch_size=batch_size,
        image_size=(ims, ims, 3),
        label_mode=None
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        batch_size=batch_size,
        image_size=(ims, ims, 3),
        label_mode=None
    )

# print(len(train_ds))
# print(len(valid_ds))
# sys.exit()

# class_names = train_ds.class_names
# num_classes = len(class_names)
# print(num_classes, class_names)
# print('train_ds len', len(train_ds))
# Check images and labels
'''plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(5):
    for i in range(6):
        ax = plt.subplot(4, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


train_arr = [images for images, labels in train_ds.take(len(train_ds))]
train_arr = np.array([elem for singleList in train_arr for elem in singleList])

valid_arr = [images for images, labels in valid_ds.take(len(valid_ds))]
valid_arr = np.array([elem for singleList in valid_arr for elem in singleList])

train_arr.astype('float32')/255.
valid_arr.astype('float32')/255.
train_arr = np.reshape(train_arr, (len(train_arr), ims, ims, 3))
valid_arr = np.reshape(valid_arr, (len(valid_arr), ims, ims, 3))
print(np.shape(train_arr), np.shape(valid_arr))
'''

data_augmentation = tf.keras.models.Sequential([
    RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    RandomRotation(0.1),
    RandomZoom(0.25)])

# #########################____MNIST DATA____##################################
if p.mnist_d:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


def fix_data(x):
    # x = x.astype(('float32', 'float32'))
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.cache()
    ds = ds.shuffle(1000, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    return ds


def create_vae():
    models = {}

    def apply_bn_and_dropout(x):
        return Dropout(dropout_r)(BatchNormalization()(x))

    # Encoder
    input_img = Input(shape=(ims, ims, 3))  # batch_shape=(batch_size, ims, ims, 3)
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
# sys.exit()


from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
disable_eager_execution()
vae_models, vae_losses = create_vae()
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

n_compare = 10

# Models
generator = vae_models["decoder"]
encoder_mean = vae_models["z_meaner"]

# print(type(valid_ds))
# print(len(valid_ds))
# sys.exit()
# imgs = valid_ds[:batch_size]  # [:batch_size] or .unbatch().take(batch_size) doesnt work
'''
# imgs = valid_ds[:batch_size]
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
'''

# lr_red = ReduceLROnPlateau(factor=0.1, patience=25)
tb = TensorBoard(log_dir=f'logs/{name}')

# Run training
# sys.exit()
if not p.use_flow_from_directory:
    for e in range(epoch):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in datagen.flow(
                os.path.join(p.dataset_folder),
                class_mode='input',
                target_size=(ims, ims),
                color_mode='rgb',
                batch_size=batch_size):
            vae.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(x_train) / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

else:
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
