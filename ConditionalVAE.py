from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten, Reshape, Lambda
from tensorflow.python.keras.layers import LeakyReLU, Concatenate
from tensorflow.keras.layers import BatchNormalization, RandomZoom, RandomRotation, RandomFlip
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import backend as K
from keras.datasets import mnist
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution

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

y_train_cat = tf.keras.utils.to_categorical(y_train).astype(np.float32)
y_test_cat = tf.keras.utils.to_categorical(y_test).astype(np.float32)
num_classes = y_test_cat.shape[1]

print('num_classes', num_classes)
print(x_train.shape[1])
print(type(x_train))
# sys.exit()

print(np.shape(x_test))
print(type(x_test))

# ######
disable_eager_execution()


def apply_bn_and_dropout(x):
    return Dropout(dropout_r)(BatchNormalization()(x))


def sample_z(args):
    mu, l_sigma = args
    eps = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return mu + K.exp(l_sigma / 2) * eps


# build model without function
inp_img = Input(shape=(x_train.shape[1],))
inp_lbl = Input(shape=(num_classes,))

inp_full = Concatenate()([inp_img, inp_lbl])
x = Dense(256, activation='relu')(inp_full)
x = apply_bn_and_dropout(x)
# predict logarithm of variation instead of standard deviation
encoded = Dense(coders_dim, activation='relu')(x)
z_mean = Dense(latent_dim, activation='linear')(x)
z_log_var = Dense(latent_dim, activation='linear')(x)

# Sampling latent space
z = Lambda(sample_z, output_shape=(latent_dim,))([z_mean, z_log_var])
# merge latent space with label
zc = Concatenate()([z, inp_lbl])

d_dens = Dense(coders_dim, activation='relu')
x = apply_bn_and_dropout(x)
x = Dense(256, activation='relu')(x)
x = apply_bn_and_dropout(x)
decoded_out = Dense(x_train.shape[1], activation='sigmoid')(x)
h_p = x(zc)
outputs = decoded_out(h_p)


def vae_loss(y_true, y_pred):
    recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    kl = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=-1)
    return recon + kl


def KL_loss(y_true, y_pred):
    return 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1)


def recon_loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


cvae = Model([inp_img, inp_lbl], outputs)
encoder = Model([inp_img, inp_lbl], z_mean)

d_in = Input(shape=(latent_dim + num_classes,))
d_h = x(d_in)
d_out = decoded_out(d_h)
decoder = Model(d_in, d_out)

cvae.compile(optimizer='adam', loss=vae_loss, metrics=[KL_loss, recon_loss])

cvae_hist = cvae.fit([x_train, y_train], x_train, verbose=1,
                     batch_size=batch_size, epochs=epoch,
                     validation_data=([x_test, y_test], x_test),
                     callbacks=[EarlyStopping(patience=5)])






