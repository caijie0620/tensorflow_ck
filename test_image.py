import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe
from datasets.dataloader import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import cv2
import numpy as np


# Path to the textfiles for the trainings and validation set
root = '/home/jie/PycharmProjects/tensorflow_ck/data/cropped_256'
train_file = 'image_lists/set1/train.txt'
val_file = 'image_lists/set1/val.txt'
test_file = 'image_lists/set1/test.txt'

# Learning params
epochs = 1
learning_rate = 0.001
learning_rate_decay = 1e-4
num_epochs = 10
batch_size = 10
dropout_rate = 0.5
num_classes = 7

MEAN = tf.constant([128, 128, 128], dtype=tf.float32)


def main():

    train_data = ImageDataGenerator(root, train_file, 'training', batch_size, num_classes, shuffle=False)

    val_data = ImageDataGenerator(root, val_file, 'inference', batch_size, num_classes)
    test_data = ImageDataGenerator(root, test_file, 'inference', batch_size, num_classes)

    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = layers.Dense(512, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=SGD(lr=learning_rate, decay=learning_rate_decay, momentum=0.9, nesterov=False),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    for e in range(epochs):
        print('Start of epoch %02d......' % (e,))
        for iter, (xs, ys) in enumerate(tfe.Iterator(train_data.dataset)):
            for j in range(len(xs)):

                image = tf.keras.backend.eval(xs[j, :, :, :])
                #cv2.imshow('image', image)
                #cv2.waitKey(0)
                image = image - image.min()
                image = image / image.max()
                cv2.imshow('image', image)
                cv2.waitKey(0)

            break





if __name__ == "__main__":
    main()
