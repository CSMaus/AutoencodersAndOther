import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

import pathlib

'''dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))
'''

'''data_dir = 'D:/DataSets/dogs_cats/'
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print('Number of images:', image_count)

batch_size = int(image_count * 0.05)
img_height = 180
img_width = 180
'''

datadir = 'D:/DataSets/dogs_cats/'


train_ds = tf.keras.utils.image_dataset_from_directory(
    datadir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(28, 28),
    batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
    datadir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(28, 28),
    batch_size=32)

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()

