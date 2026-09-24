"""Small CRNN sized for real-time inference on a Pi 4, trained from scratch."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_crnn(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="logmel")

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = layers.Dropout(0.3)(x)

    # x is (batch, mel, time, channels) -- swap to (batch, time, mel, channels)
    # then collapse mel*channels into one feature axis so the RNN steps over time
    x = layers.Permute((2, 1, 3))(x)
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    # unroll=True: sequence length is fixed (set by CLIP_SECONDS), so the RNN
    # compiles to static ops instead of TensorList ops -- the latter can't be
    # lowered by the standard TFLite converter without the Flex delegate,
    # which isn't practical to get working on the Pi in this timeframe.
    x = layers.Bidirectional(layers.GRU(24, unroll=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="intent")(x)

    return keras.Model(inputs, outputs, name="vcm_crnn")
