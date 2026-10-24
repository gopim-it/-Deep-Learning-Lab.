"""Task 12: Denoising autoencoder on MNIST handwritten digit images."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

np.random.seed(7)
tf.random.set_seed(7)
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32")[..., None] / 255.0
x_test = x_test.astype("float32")[..., None] / 255.0
noise_train = np.clip(x_train + 0.35 * np.random.normal(size=x_train.shape), 0, 1).astype("float32")
noise_test = np.clip(x_test + 0.35 * np.random.normal(size=x_test.shape), 0, 1).astype("float32")

inputs = layers.Input((28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(2, padding="same")(x)
x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)
x = layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(noise_train, x_train, epochs=3, batch_size=128,
          validation_split=0.1, verbose=2)
reconstructed = model.predict(noise_test[:10], verbose=0)
np.save("denoised_images.npy", reconstructed)
model.save("denoising_autoencoder.keras")
print("Saved denoised images to denoised_images.npy")
print("Reconstruction array shape:", reconstructed.shape)
