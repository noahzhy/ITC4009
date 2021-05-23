import os
import PIL
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def loading_dataset(ds="fashion", bs=32):
    if ds == "fashion":
        num_classes = 10
        fashion_mnist = keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        x_train = (np.expand_dims(x_train, axis=3))
        x_test = (np.expand_dims(x_test, axis=3))

        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        train_ds = ImageDataGenerator().flow(
            x_train, y_train, batch_size=bs
        )
        val_ds = ImageDataGenerator().flow(
            x_test, y_test, batch_size=bs
        )

    elif ds == "flower":
        data_path = download_dataset()
        train_ds, val_ds = data_preprocessing(data_path, ds)

    return train_ds, val_ds


def download_dataset():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(
        'flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    return data_dir


def data_preprocessing(data_dir, ds, bs=32):
    if ds == "fashion":
        image_size = (28, 28)
    elif ds == "flower":
        image_size = (180, 180)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=bs
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=bs
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def data_augmentation(input_shape=(28, 28, 1), aug=True):
    if aug:
        input_layer = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip(
                    "horizontal", input_shape=input_shape),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.1),
            ]
        )

    else:
        input_layer = layers.Input(shape=input_shape)

    return input_layer


def build_model(data_aug=True, model_desc="fashion", input_shape=(28, 28, 1), num_classes=10):
    model = Sequential()
    # input_shape in data_augmentation
    model.add(data_augmentation(input_shape, data_aug))
    model.add(layers.experimental.preprocessing.Rescaling(1./255))

    if model_desc == "fashion":
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))

    elif model_desc == "flower":
        model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(num_classes))
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()
    return model


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


def predict():
    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file(
        'Red_sunflower', origin=sunflower_url)

    img = keras.preprocessing.image.load_img(
        sunflower_path, target_size=(180, 180)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


def train(train_ds, val_ds, model, epochs=20):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return history


if __name__ == "__main__":
    num_classes_fashion = 10
    num_classes_flower = 5

    batch_size = 32
    validation_split = 0.2
    epochs = 15
    seed = 123

    # # E-01
    # train_ds, test_ds = loading_dataset(ds="flower")
    # model = build_model(
    #     data_aug=False,
    #     model_desc="fashion",
    #     input_shape=(180, 180, 3),
    #     num_classes=num_classes_flower
    # )
    # train(train_ds, test_ds, model, epochs=epochs)

    # E-02
    train_ds, test_ds = loading_dataset(ds="fashion")
    model = build_model(
        data_aug=True,
        model_desc="flower",
        input_shape=(28, 28, 1),
        num_classes=num_classes_fashion
    )
    train(train_ds, test_ds, model, epochs=epochs)

    # results = train(train_ds, val_ds, 15)
    # visualize_results(results)
    # predict()
