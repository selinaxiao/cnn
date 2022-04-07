import numpy
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import glob
import Code
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from PIL import Image

output = glob.glob("arcDataset/*/*.jpg")
# print(output)

t_lower = 300
t_upper = 350


def show_image(image, title):
    plt.imshow(image, cmap=cm.gray)
    plt.title(title)
    plt.show()


result = []

target_height = 300
target_width = 400

image_input = []
for filename in output:
    img = cv2.imread(filename)
    # print(np.shape(img))
    img1 = tf.image.resize_with_crop_or_pad(img, target_height, target_width)
    # img1 = tf.image.rgb_to_grayscale(img1, name=None)
    image_input.append(img1)

    # print(type(img1.numpy()), '\n', type(img))
    out = cv2.Canny(img1.numpy(), t_lower, t_upper)
    # print(np.shape(out))
    # show_image(out, 'Final Image for'+filename)
    # result.append(tf.image.resize_with_crop_or_pad(Image.fromarray(out), target_height, target_width))
    result.append(np.ndarray.flatten(out))

image_train, image_test, labels_train, labels_test = train_test_split(image_input, result, test_size=0.20,
                                                                      random_state=42)

# show_image(image_train[0], 'Original Image')
# show_image(labels_train[0].reshape((target_height, target_width)), 'Final Image')

normalization_layer = layers.Rescaling(1. / 255)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_train[i])
# plt.show()


model = models.Sequential()
# conv1,BN, Relu
model.add(layers.Conv3D(64, (3, 3, 4), padding='same', input_shape=(13, 13, 4)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

# conv2,BN, Relu,Pool
model.add(layers.Conv3D(128, (3, 3, 64), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

# conv3,BN, Relu,Pool
model.add(layers.Conv3D(256, (3, 3, 128), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

# conv4,BN, Relu
model.add(layers.Conv3D(512, (3, 3, 256), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

# conv5,BN, Relu
model.add(layers.Conv3D(1024, (3, 3, 512), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

# FC, output 1
model.add(layers.Flatten())
model.add(layers.Dense(1))

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

image_train = np.array(image_train)
labels_train = np.array(labels_train)

image_test = np.array(image_test)
labels_test = np.array(labels_test)

print(image_train.shape, labels_train.shape)

"""

history = model.fit(image_train, labels_train, epochs=10,
                    validation_data=(image_test, labels_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(image_test, labels_test, verbose=2)

print(test_acc)
"""
