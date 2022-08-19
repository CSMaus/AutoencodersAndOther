from typing import List, Tuple
import tensorflow as tf
import random
AUTOTUNE = tf.data.experimental.AUTOTUNE


class MyDataLoader(object):
    """A TensorFlow Dataset API based loader for semantic segmentation problems."""

    def __init__(self, image_paths: List[str], mask_paths: List[str], image_size: Tuple[int],
                 channels: Tuple[int] = (3, 1), seed: int = None,
                 augment: bool = True, compose: bool = False, one_hot_encoding: bool = False, palette=None):
        """
        Initializes the data loader object
        Args:
            image_paths: List of paths of train images.
            mask_paths: List of paths of train masks (segmentation masks)
            image_size: Tuple, the final height, width of the loaded images.
            channels: Tuple of ints, first element is number of channels in images,
                      second is the number of channels in the mask image (needed to
                      correctly read the images into tensorflow and apply augmentations)
            crop_percent: Float in the range 0-1, defining percentage of image
                          to randomly crop.
            palette: A list of RGB pixel values in the mask. If specified, the mask
                     will be one hot encoded along the channel dimension.
            seed: An int, if not specified, chosen randomly. Used as the seed for
                  the RNG in the data pipeline.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.palette = palette
        self.image_size = image_size
        self.augment = augment
        self.compose = compose
        self.one_hot_encoding = one_hot_encoding
        self.channels = channels

    def _resize_data(self, image, mask):
        """
        Resizes images to specified size.
        """
        image = tf.image.resize(image, self.image_size)
        mask = tf.image.resize(mask, self.image_size)  # , method="nearest"

        return image, mask

    def _parse_data(self, image_paths, mask_paths):
        """
        Reads image and mask files depending on
        specified exxtension.
        """
        image_content = tf.io.read_file(image_paths)
        mask_content = tf.io.read_file(mask_paths)

        images = tf.image.decode_jpeg(image_content, channels=self.channels[0])
        masks = tf.image.decode_jpeg(mask_content, channels=self.channels[1])

        return images, masks

    def _flip_left_right(self, image, mask):
        """
        Randomly flips image and mask left or right in accord.
        """
        comb_tensor = tf.concat([image, mask], axis=2)
        comb_tensor = tf.image.random_flip_left_right(comb_tensor, seed=self.seed)
        image, mask = tf.split(comb_tensor, [self.channels[0], self.channels[1]], axis=2)

        return image, mask

    def _resize_data(self, image, mask):
        """
        Resizes images to specified size.
        """
        image = tf.image.resize(image, self.image_size)
        mask = tf.image.resize(mask, self.image_size)

        return image, mask

    def _parse_data(self, image_paths, mask_paths):
        """
        Reads image and mask files depending on
        specified exxtension.
        """
        image_content = tf.io.read_file(image_paths)
        mask_content = tf.io.read_file(mask_paths)

        images = tf.image.decode_jpeg(image_content, channels=self.channels[0])
        masks = tf.image.decode_jpeg(mask_content, channels=self.channels[1])

        return images, masks

    @tf.function
    def _map_function(self, images_path, masks_path):
        image, mask = self._parse_data(images_path, masks_path)

        def _augmentation_func(image_f, mask_f):

            image_f, mask_f = self._resize_data(image_f, mask_f)
            return image_f, mask_f

        return tf.py_function(_augmentation_func, [image, mask], [tf.uint8, tf.uint8])  #  [tf.float32, tf.uint8]

    def data_batch(self, batch_size, shuffle=False):
        """
        Reads data, normalizes it, shuffles it, then batches it, returns a
        the next element in dataset op and the dataset initializer op.
        Inputs:
            batch_size: Number of images/masks in each batch returned.
            augment: Boolean, whether to augment data or not.
            shuffle: Boolean, whether to shuffle data in buffer or not.
            one_hot_encode: Boolean, whether to one hot encode the mask image or not.
                            Encoding will done according to the palette specified when
                            initializing the object.
        Returns:
            data: A tf dataset object.
        """

        # Create dataset out of the 2 files:
        data = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))

        # Parse images and labels
        data = data.map(self._map_function, num_parallel_calls=AUTOTUNE)

        if shuffle:
            # Prefetch, shuffle then batch
            data = data.prefetch(AUTOTUNE).shuffle(random.randint(0, len(self.image_paths))).batch(batch_size)
        else:
            # Batch and prefetch
            data = data.batch(batch_size).prefetch(AUTOTUNE)

        return data


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
























