import keras.optimizers
import numpy as np, os, pathlib, matplotlib.pyplot as plt, sys, seaborn as sns
from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, Reshape, Lambda, Concatenate
from keras.models import Model
from keras.layers import concatenate
from keras.layers import Rescaling, Reshape, Resizing, RandomZoom, RandomRotation, RandomFlip

import tensorflow as tf
from keras.metrics.metrics import binary_crossentropy
from keras.layers import LeakyReLU
from keras import backend as bk
# from tensorflow.keras.utils import to_categorical
import parameters as p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# PARAMETERS
batch_size = p.batch_size
latent_dim = p.latent_dim  # to be easier generate and visualize result
dropout_r = p.dropout_r
lr_0 = p.lr_0
epoch = p.epoch

img_height = p.img_size
img_width = p.img_size

Adam = keras.optimizers.Adam

# my data
tf.debugging.set_log_device_placement(True)
list_gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in list_gpu:
    tf.config.experimental.set_memory_growth(gpu, True)

ims = p.img_size
name = f'pets_cvae_dim{latent_dim}_epochs{epoch}_ims{ims}'

data_dir = p.dataset_folder
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print('Number of images:', image_count)

# if batch_size < image_count:
#     batch_size = int(image_count * 0.05)
#     # print(batch_size)
# else:
#     batch_size = image_count // 2

print('BATCH SIZE:', batch_size)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    batch_size=batch_size,
    image_size=(ims, ims),
    labels='inferred',
    label_mode='categorical',
    color_mode="rgb"
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    batch_size=batch_size,
    image_size=(ims, ims),
    labels='inferred',
    label_mode='categorical',
    color_mode="rgb"
)




ishape = (ims, ims, 3)
class_names = train_ds.class_names  # os.listdir(os.path.join(data_dir))
num_classes = len(class_names)
print('num_classes:', num_classes, '-', class_names)


full_imgs = lambda ds: np.concatenate([x for x, y in ds])
full_lbls = lambda ds: np.concatenate([y for x, y in ds])

train_img = full_imgs(train_ds)
train_img = train_img.astype('float32') / 255.
train_lbl = full_lbls(train_ds)
# train_lbl_cat = tf.keras.utils.to_categorical(train_lbl).astype(np.float32)

valid_img = full_imgs(valid_ds)
valid_img = valid_img.astype('float32') / 255.
valid_lbl = full_lbls(valid_ds)
# valid_lbl_cat = tf.keras.utils.to_categorical(valid_lbl).astype(np.float32)

print('\n\nshape train_img, train_lbl:', np.shape(train_img), np.shape(train_lbl), '\n')

# train_lbls = train_lbl[0]
# print(np.shape(train_lbl.T[0]))
# sys.exit()

'''
train_ds = image_generator.flow_from_directory(
    os.path.join(p.dataset_folder),
    # class_mode='input',
    target_size=(ims, ims),
    batch_size=batch_size,
    subset="training",
    color_mode='rgb',
    label_mode='categorical'
)

valid_ds = image_generator.flow_from_directory(
    os.path.join(p.dataset_folder),
    # class_mode='input',
    target_size=(ims, ims),
    batch_size=batch_size,
    subset="validation",
    color_mode='rgb',
    label_mode='categorical'
)'''

data_augmentation = tf.keras.models.Sequential([
    RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    RandomRotation(0.1),
    RandomZoom(0.25)])
# sys.exit()

# #######################################################################################
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
disable_eager_execution()
# #######################################################################################


# remake to conditional VAE for pets images
def create_cvae():
    models = {}

    def apply_bn_and_dropout(x):
        return Dropout(dropout_r)(BatchNormalization()(x))

    # Encoder
    inp_img = Input(shape=ishape)  # batch_shape=(batch_size, ims, ims, 1)
    flat = Flatten()(inp_img)
    inp_lbls = Input(shape=(num_classes,), dtype='float32')

    x = concatenate([flat, inp_lbls])
    x = Dense(1024, activation='relu')(x)
    x = apply_bn_and_dropout(x)
    x = Dense(512, activation='relu')(x)
    # x = apply_bn_and_dropout(x)

    # predict logarithm of variation instead of standard deviation
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    # sampling from Q with reparametrisation
    def sampling(args):
        z_means, z_log_vars = args
        epsilon = bk.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_means + bk.exp(z_log_vars / 2) * epsilon

    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    l_z = concatenate([l, inp_lbls])

    encoder = Model([inp_img, inp_lbls], l_z, name='my_encoder')
    z_meaner = Model([inp_img, inp_lbls], z_mean, name='Enc_z_mean')
    models["encoder"] = encoder
    models["z_meaner"] = z_meaner
    models["z_lvarer"] = Model([inp_img, inp_lbls], z_log_var, name='Enc_z_log_var')

    # Decoder
    z = Input(shape=(latent_dim + num_classes,))
    x = Dense(512)(z)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(1024)(z)
    x = LeakyReLU()(x)
    # x = apply_bn_and_dropout(x)
    x = Dense(ims * ims * 3, activation='sigmoid')(x)
    decoded = Reshape(ishape)(x)

    decoder = Model(z, decoded, name='my_decoder')  # [z, inp_lbls_d]

    models["decoder"] = decoder

    cvae_out = decoder(encoder([inp_img, inp_lbls]))

    my_cvae = Model([inp_img, inp_lbls], cvae_out, name='my_cvae')
    models['cvae'] = my_cvae

    out_style = decoder(concatenate([z_meaner([inp_img, inp_lbls]), inp_lbls]))
    models["style_t"] = Model([inp_img, inp_lbls], out_style, name="style_transfer")

    def vae_loss(x, decoded):
        x = bk.reshape(x, shape=(batch_size, ims * ims * 3))
        decoded = bk.reshape(decoded, shape=(batch_size, ims * ims * 3))
        xent_loss = ims * ims * 3 * binary_crossentropy(x, decoded)
        kl_loss = -0.5 * bk.sum(1 + z_log_var - bk.square(z_mean) - bk.exp(z_log_var), axis=-1)
        return (xent_loss + kl_loss)/2/ims/ims/3

    return models, vae_loss


from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

cvae_models, cvae_losses = create_cvae()
cvae = cvae_models["cvae"]

cvae.compile(optimizer='adam', loss=cvae_losses, experimental_run_tf_function=False)  # cvae_losses

# Plot images / imagss
imags_size = ims


def plot_imagss(*args, invert_colors=False):
    args = [x.squeeze() for x in args]
    n_f = min([x.shape[0] for x in args])
    figure = np.zeros((imags_size * len(args), imags_size * n_f, 3))

    for i in range(n_f):
        for j in range(len(args)):
            figure[j * imags_size: (j + 1) * imags_size,
            i * imags_size: (i + 1) * imags_size, :] = args[j][i].squeeze()

    if invert_colors:
        figure = 1 - figure

    plt.figure(figsize=(2 * n_f, 2 * len(args)))
    plt.imshow(figure)  # cmap='Greys_r'
    plt.grid(False)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


n = 15  # Img with 15x15 imagss

from scipy.stats import norm

# Since we are sampling from N(0, I),
# we take the grid of nodes in which we generate numbers from the inverse distribution function
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))


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
imgs = valid_img[:batch_size]
imgs_lbls = valid_lbl[:batch_size]
n_compare = 10

# Models
generator = cvae_models["decoder"]
encoder_mean = cvae_models["z_meaner"]


# The function that we will run after each epoch
def on_epoch_end(epoch, logs):
    if epoch in save_epochs:
        clear_output()

        # Comparison of real and decoded numbers
        decoded = cvae.predict([imgs, imgs_lbls, imgs_lbls], batch_size=batch_size)
        plot_imagss(imgs[:n_compare], decoded[:n_compare])

        draw_lbl = np.random.randint(0, num_classes)
        print(draw_lbl)
        for lbl in range(num_classes):
            figs[lbl].append(draw_manifold(generator, lbl, show=lbl == draw_lbl))
            idxs = valid_img == lbl
            z_predicted = encoder_mean.predict([valid_img[idxs], imgs_lbls[idxs]], batch_size)
            latent_distrs[lbl].append(z_predicted)
            if lbl == draw_lbl:
                draw_z_distr(z_predicted, lbl)
        epochs.append(epoch)


# Callback
lambda_pltfig = LambdaCallback(on_epoch_end=on_epoch_end)

# lr_red = ReduceLROnPlateau(factor=0.1, patience=25)
tb = TensorBoard(log_dir=f'logs/{name}')

# Run training
cvae.fit(
    x=[train_img, train_lbl],
    y=train_img,
    batch_size=batch_size,
    shuffle=True,
    epochs=epoch,
    validation_data=([valid_img, valid_lbl], valid_img),
    callbacks=[tb],
    verbose=1)


def style_transfer(model, X, lbl_in, lbl_out):
    rows = X.shape[0]
    if isinstance(lbl_in, int):
        lbl_f = lbl_in
        lbl_in = np.zeros((rows, 2))
        lbl_in[:, lbl_f] = 1
    if isinstance(lbl_out, int):
        lbl_f = lbl_out
        lbl_out = np.zeros((rows, 2))
        lbl_out[:, lbl_f] = 1
    return model.predict([X, lbl_in, lbl_out])


n = 5
# lbls = [2, 3, 5, 6, 7]
# for lbl in lbls:
lbl = 1
generated = []
# prot = x_train[y_train == lbl][:n]


for i in range(num_classes):
    print(i)
    prot = train_img[train_lbl.T[0] == i][:n]
    generated.append(style_transfer(cvae_models["style_t"], prot, lbl, i))

prot = train_img[train_lbl.T[0] == lbl][:n]
generated[lbl] = prot
plot_imagss(*generated, invert_colors=False)

# sys.exit()

# Comparison of real and decoded numbers
print(type(imgs))
print(imgs.shape)
decoded = cvae.predict([imgs, imgs_lbls], batch_size=batch_size)
plot_imagss(imgs[:n_compare], decoded[:n_compare])
