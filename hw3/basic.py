import os
import PIL
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


def visualize_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def loading_dataset():
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = (np.expand_dims(x_train, -1))
    x_test = (np.expand_dims(x_test, -1))

    train_ds = ImageDataGenerator().flow(
        x_train, y_train, batch_size=32
    )
    val_ds = ImageDataGenerator().flow(
        x_test, y_test, batch_size=32
    )

    return train_ds, val_ds


def train(train_ds, val_ds, epochs=20):
    model = Sequential()
    model.add(layers.Input(shape=(28, 28)))
    model.add(layers.experimental.preprocessing.Rescaling(1./255))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes))

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return history


if __name__ == "__main__":
    image_size = (28, 28)
    num_classes = 10
    batch_size = 32
    validation_split = 0.2
    epochs = 20
    seed = 123
    # E-01
    train_ds, test_ds = loading_dataset()
    results = train(train_ds, test_ds, epochs=epochs)

    visualize_results(results)
