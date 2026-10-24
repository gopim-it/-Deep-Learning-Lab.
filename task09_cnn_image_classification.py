"""Task 9: CNN image classification using CIFAR-10."""
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
y_train, y_test = y_train.squeeze(), y_test.squeeze()
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

model = models.Sequential([
    layers.Input((32, 32, 3)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
pred = int(model.predict(x_test[:1], verbose=0).argmax(axis=1)[0])
print(f"Test accuracy: {acc:.4f}")
print(f"Actual class: {classes[int(y_test[0])]}; predicted class: {classes[pred]}")
