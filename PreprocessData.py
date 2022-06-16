import numpy as np, sys, os, matplotlib as plt
import parameters as p

# PARAMETERS
batch_size = p.batch_size
latent_dim = p.latent_dim  # to be easier generate and visualize result
dropout_r = p.dropout_r
lr_0 = p.lr_0
epoch = p.epoch

img_height = p.img_size
img_width = p.img_size

if p.img_as_gray:
    img_shape = (img_width, img_height)
else:
    img_shape = (img_width, img_height, 3)

data_dir = p.dataset_folder

