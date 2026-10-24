"""Task 8: Binary classification on a generated two-class dataset."""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from tensorflow.keras import layers, models

np.random.seed(7)
tf.random.set_seed(7)
X, y = make_moons(n_samples=2000, noise=0.2, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(
    X.astype("float32"), y.astype("float32"), test_size=0.2, random_state=7)

model = models.Sequential([
    layers.Input((2,)), layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"), layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=0)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")
print("Predicted class for first test sample:",
      int(model.predict(X_test[:1], verbose=0)[0, 0] >= 0.5))
