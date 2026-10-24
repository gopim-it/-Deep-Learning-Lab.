"""Task 1: MNIST handwritten digit classification using a CNN."""
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., None].astype("float32") / 255.0
x_test = x_test[..., None].astype("float32") / 255.0

model = models.Sequential([
    layers.Input((28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
pred = int(model.predict(x_test[:1], verbose=0).argmax(axis=1)[0])
print(f"Test accuracy: {accuracy:.4f}")
print(f"Actual digit: {int(y_test[0])}; predicted digit: {pred}")
