import numpy as np, sys, os, matplotlib as plt
import parameters as p
from tqdm import tqdm
from PIL import Image
from copy import copy
from random import randint

# PARAMETERS
batch_size = p.batch_size
img_height = p.img_size
img_width = p.img_size

if p.img_as_gray:
    img_shape = (img_width, img_height, 1)
else:
    img_shape = (img_width, img_height, 3)

data_dir = p.dataset_folder

train_ds = []
valid_ds = []
test_ds = []


def fill_ds_from_splitted_folders(
        ds_folder,
        ds_type,
        img_w=224,
        img_h=224,
        color_rgb=True,
        flip_y=False,
        flip_x=False):
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
    num_categories = len(categories)

    for category in categories:
        category_folder = f'{ds_folder}/{ds_type}/{category}'
        num_of_ims = len(os.listdir(os.path.join(category_folder)))

        for img in tqdm(os.listdir(os.path.join(category_folder))):
            # try:
            if color_rgb:
                img_arr = np.array(Image.open(os.path.join(category_folder, img)).convert('RGB').resize((img_width, img_height), Image.ANTIALIAS))
                out_ds.append(img_arr)
            else:
                img_arr = Image.open(os.path.join(category_folder, img)).convert('L')
                img_arr = img_arr.resize([img_width, img_height])
                out_ds.append(np.array(img_arr))
                img_arr = np.expand_dims(img_arr, axis=2)
                out_ds.append(img_arr)

            # if flip_y:
                # flipped_y = copy(cv2.flip(img_arr, 1))
                # flipped_y = np.expand_dims(flipped_y, axis=2)
                # if not color_rgb:
                #     out_ds.append(flipped_y)

            # if flip_x:
                # flipped_x = copy(cv2.flip(img_arr, 0))
                # flipped_x = np.expand_dims(flipped_x, axis=2)
                # if not color_rgb:
                #     out_ds.append(flipped_x)

            # except:
            #     pass

            # print(len(out_ds))
            # print(np.shape(out_ds))
            # if (num_of_ims-len(out_ds)) * num_categories == batch_size:
            #     break
            # print(np.shape(out_ds))
            # print('len(out_ds)', len(out_ds))
            # break
        # break
    # print(np.shape(out_ds))
    return np.array(out_ds)


# remove elements to make it possible to split the array into 100 batches
def remove_equal_batches(arr, BS=100):
    while np.shape(arr)[0] % BS != 0:
        arr = np.delete(arr, randint(0, np.shape(arr)[0] - 2), 0)
        # print(np.shape(arr)[0])
    return arr


def do_prepare(
        ds_folder,
        ds_type,
        batch_s,
        img_w=224,
        img_h=224,
        color_rgb=True,
        flip_y=False,
        flip_x=False):
    out_array = fill_ds_from_splitted_folders(
        ds_folder,
        ds_type,
        img_w=img_w,
        img_h=img_h,
        color_rgb=color_rgb,
        flip_y=flip_y,
        flip_x=flip_x
    )
    return remove_equal_batches(out_array, batch_s)


try_test_ds = fill_ds_from_splitted_folders(data_dir, ds_type='train')
# try_test_ds = np.array(try_test_ds)

# x = try_test_ds[0]
print(np.shape(try_test_ds))
print(type(try_test_ds))

sys.exit()
try_test_ds = np.delete(try_test_ds, randint(0, np.shape(try_test_ds)[0] - 2), 0)
try_test_ds = remove_equal_batches(try_test_ds)
print(np.shape(try_test_ds))
print(type(try_test_ds))
