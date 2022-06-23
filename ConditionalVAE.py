'''from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten, Reshape, Lambda
from tensorflow.python.keras.layers import LeakyReLU, Concatenate
from tensorflow.keras.layers import BatchNormalization, RandomZoom, RandomRotation, RandomFlip
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import backend as K
from keras.datasets import mnist
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
'''
from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, Reshape, Lambda, LayerNormalization
from keras.models import Model, Sequential
from keras.layers import LeakyReLU, Concatenate
from keras.datasets import mnist
from keras.layers import concatenate
from keras.metrics.metrics import binary_crossentropy
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers import Rescaling, Reshape, Resizing, RandomZoom, RandomRotation, RandomFlip


import pathlib
import numpy as np, matplotlib.pyplot as plt, sys, os
import parameters as p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# PARAMETERS
batch_size = p.batch_size
latent_dim = p.latent_dim  # to be easier generate and visualize result
coders_dim = 512
dropout_r = p.dropout_r
lr_0 = p.lr_0
epoch = p.epoch
opt = Adam(lr=0.001)

img_height = p.img_size
img_width = p.img_size

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

n_pixels = np.prod(x_train.shape[1:])
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_train = np.reshape(x_train, (len(x_train), n_pixels))
x_test = np.reshape(x_test, (len(x_test), n_pixels))

y_train = tf.keras.utils.to_categorical(y_train).astype(np.float32)
y_test = tf.keras.utils.to_categorical(y_test).astype(np.float32)
num_classes = y_test.shape[1]

print('num_classes', num_classes)
print(x_train.shape)
print(type(x_train))
# sys.exit()

print(np.shape(y_train))
print(type(y_train))
# sys.exit()
# ######
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def apply_bn_and_dropout(x):
    return Dropout(dropout_r)(LayerNormalization()(x))


def sample_z(args):
    mu, l_sigma = args
    eps = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return mu + K.exp(l_sigma / 2) * eps


# build model without function
inp_img = Input(shape=(n_pixels,))
inp_lbl = Input(shape=(num_classes,))

inp_full = concatenate([inp_img, inp_lbl])  # ([inp_img, inp_lbl])
x = Dense(256, activation='relu')(inp_full)
x = apply_bn_and_dropout(x)
# predict logarithm of variation instead of standard deviation
encoded = Dense(coders_dim, activation='relu')(x)
z_mean = Dense(latent_dim, activation='linear')(encoded)
z_log_var = Dense(latent_dim, activation='linear')(encoded)

# Sampling latent space
z = Lambda(sample_z, output_shape=(latent_dim,))([z_mean, z_log_var])
# merge latent space with label
zc = concatenate([z, inp_lbl])  # ([z, inp_lbl])


'''
def decoded_hidden(inp_layer):
    # inp_ = inp_layer
    lr = Dense(coders_dim, activation='relu')(inp_layer)
    # lr = Dropout(dropout_r)(lr),
    lr = apply_bn_and_dropout(lr),
    lr = Dense(256, activation='relu')(lr)
    return lr'''


decoded_hidden = Dense(coders_dim, activation='relu')
decoded_out = Dense(n_pixels, activation='sigmoid')
dec_h = decoded_hidden(zc)
dec_out = decoded_out(dec_h)


def vae_loss(y_true, y_pred):
    recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    kl = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=-1)
    return recon + kl


def KL_loss(y_true, y_pred):
    return 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1)


def recon_loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


# ##################################

d_in = Input(shape=(latent_dim + num_classes,))
d_h = Dense(coders_dim, activation='relu')(d_in)
x = apply_bn_and_dropout(d_h)
x = Dense(256, activation='relu')(x)
# x = LeakyReLU()(x)
d_out = Dense(n_pixels, activation='sigmoid')(x)

# d_h = decoded_hidden(x)
# d_out = decoded_out(d_h)

# cvae = Model([inp_img, inp_lbl], dec_out)
encoder = Model([inp_img, inp_lbl], z_mean)

decoder = Model(d_in, d_out)

cvae_out = decoder(concatenate([encoder([inp_img, inp_lbl]), inp_lbl]))
cvae = Model([inp_img, inp_lbl], cvae_out)

cvae.compile(optimizer=opt, loss=vae_loss, metrics=[KL_loss, recon_loss])


import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

cvae_hist = cvae.fit([x_train, y_train], x_train, verbose=1,
                     batch_size=batch_size, epochs=epoch,
                     validation_data=([x_test, y_test], x_test),
                     callbacks=[EarlyStopping(patience=5)])


def construct_numvec(digit, zf=None):
    out = np.zeros((1, latent_dim + num_classes))
    out[:, digit + latent_dim] = 1.
    if zf is None:
        return (out)
    else:
        for i in range(len(zf)):
            out[:, i] = zf[i]
        return (out)


sample_3 = construct_numvec(3)
print(sample_3)

plt.figure(figsize=(3, 3))
# plt.imshow(decoder.predict(sample_3).reshape(28, 28), cmap='gray')
plt.imshow(decoder.predict(sample_3).reshape(28, 28), cmap='gray')
plt.show()
sys.exit()

dig = 4
sides = 8
max_z = 1.5

img_it = 0
for i in range(0, sides):
    z1 = (((i / (sides-1)) * max_z)*2) - max_z
    for j in range(0, sides):
        z2 = (((j / (sides-1)) * max_z)*2) - max_z
        z_ = [z1, z2]
        vec = construct_numvec(dig, z_)
        decoded = decoder.predict(vec)
        plt.subplot(sides, sides, 1 + img_it)
        img_it += 1
        plt.imshow(decoded.reshape(28, 28), cmap='gray')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)
plt.show()

dig = 2
sides = 8
max_z = 1.5

img_it = 0
for i in range(0, sides):
    z1 = (((i / (sides-1)) * max_z)*2) - max_z
    for j in range(0, sides):
        z2 = (((j / (sides-1)) * max_z)*2) - max_z
        z_ = [z1, z2]
        vec = construct_numvec(dig, z_)
        decoded = decoder.predict(vec)
        plt.subplot(sides, sides, 1 + img_it)
        img_it += 1
        plt.imshow(decoded.reshape(28, 28), cmap='gray')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)
plt.show()


dig = 6
sides = 8
max_z = 1.5

img_it = 0
for i in range(0, sides):
    z1 = (((i / (sides-1)) * max_z)*2) - max_z
    for j in range(0, sides):
        z2 = (((j / (sides-1)) * max_z)*2) - max_z
        z_ = [z1, z2]
        vec = construct_numvec(dig, z_)
        decoded = decoder.predict(vec)
        plt.subplot(sides, sides, 1 + img_it)
        img_it += 1
        plt.imshow(decoded.reshape(28, 28), cmap='gray')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)
plt.show()

