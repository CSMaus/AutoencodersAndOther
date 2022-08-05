import numpy as np, os, sys, matplotlib.pyplot as plt, pathlib
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
import datetime
import shutup
shutup.please()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# if GPU exist and cudnn and cudart downloaded for GPU
# tf.debugging.set_log_device_placement(True)
# list_gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in list_gpu:
#     tf.config.experimental.set_memory_growth(gpu, True)

# python -m tensorboard.main --logdir="logs/"

train_data_dir = 'sorted/train_valid/'
test_data_dir = 'sorted/test/'

train_data_dir = pathlib.Path(train_data_dir)
test_data_dir = pathlib.Path(test_data_dir)

# Data load and dataset creation
batch_size = 24
img_height = 512
img_width = 512
img_size = (img_height, img_width)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_dataset.class_names
num_classes = len(class_names)


plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

val_batches = tf.data.experimental.cardinality(validation_dataset)
# test_dataset = validation_dataset.take(val_batches // 5)
# validation_dataset = validation_dataset.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
# print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
# test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')

rescale = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)
# Create the base model from the pre-trained model MobileNet V2
img_shape = img_size + (3,)

# ################ THIS WAS FOR GPU CALCULATIONS
# tf.debugging.set_log_device_placement(True)
# logical_gpus = tf.config.list_logical_devices('GPU')
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=logical_gpus,
#                                                    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
# with mirrored_strategy.scope():

# ################################# HERE IMPORT DIFFERENT nn AND TEST THEM ####################################
import tensorflow.keras.applications as app
# base_model = app.InceptionResNetV2(
base_model = app.vgg16.VGG16(
    # base_model = app.efficientnet.EfficientNetB4(
    input_shape=img_shape,
    include_top=False,
    weights='imagenet')

name = f'VGG16-based_50categories_classification_im_size={img_height}__'

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

# base model architecture (summary):
base_model.summary()
print(type(base_model))
print(base_model.layers[-1].output)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(num_classes)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=img_shape)
x = data_augmentation(inputs)
x = rescale(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0002
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# TL model summary
model.summary()
with open(f'ModelSummaries/{name}_summary.txt', 'w+') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
# sys.exit()
loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(100 * loss0))
print("initial accuracy: {:5.2f}%".format(100 * accuracy0))

checkpoint_path = f'checkpoint_path/{name}/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callbacks that save model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq='epoch')
model.save_weights(checkpoint_path.format(epoch=0))
# train the model
tensorboard = TensorBoard(log_dir='logs/{}'.format(name) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    shuffle=True,
    callbacks=[tensorboard])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
