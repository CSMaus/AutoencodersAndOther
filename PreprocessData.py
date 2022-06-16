import numpy as np, sys, os, matplotlib as plt
import parameters as p
from tqdm import tqdm
import cv2
from copy import copy

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

train_ds = []
valid_ds = []
test_ds = []


def fill_ds(ds_folder, ds_type='train', img_w=224, img_h=224, color_rgb=False, flip_y=True, flip_x=True):
    """
    :param ds_folder: folder with train, validation and test datasets
    :param ds_type: choose 'train', 'validation' or 'test'
    :param img_h: image height to resize
    :param img_w: image width to resize
    :param color_rgb: color mode - if True then 3 color channels, else 1 color channel
    :param flip_y: add image flipped around y
    :param flip_x: add image flipped around x
    :return: filled dataset of resized images
    """
    out_ds = []
    categories = os.listdir((os.path.join(f'{ds_folder}/{ds_type}')))

    for category in categories:
        category_folder = f'{ds_folder}/{ds_type}/{category}'

        for img in tqdm(os.listdir(os.path.join(f'{ds_folder}/{ds_type}', category))):
            # try:
            if color_rgb:
                img_arr = cv2.imread(os.path.join(category_folder, img), cv2.COLOR_BGR2RGB)
                img_arr = cv2.resize(img_arr, (img_w, img_h))
                out_ds.append(img_arr)
            else:
                img_arr = cv2.imread(os.path.join(category_folder, img), cv2.IMREAD_GRAYSCALE)
                img_arr = cv2.resize(img_arr, (img_w, img_h))
                out_ds.append(np.expand_dims(img_arr, axis=2))

            if flip_y:
                flipped_y = copy(cv2.flip(img_arr, 1))
                out_ds.append(np.expand_dims(flipped_y, axis=2))

            if flip_x:
                flipped_x = copy(cv2.flip(img_arr, 0))
                out_ds.append(np.expand_dims(flipped_x, axis=2))

                # print(np.shape(img_arr))

            # except:
            #     pass

            break

        # break
    return np.asarray(out_ds, dtype=np.float32)


try_test_ds = fill_ds(data_dir)
print(np.shape(try_test_ds))
print(type(try_test_ds))


