"""Task 5: Forecast a synthetic time series with an LSTM."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

np.random.seed(7)
tf.random.set_seed(7)
t = np.arange(0, 1500)
series = (np.sin(t * 0.03) + 0.3 * np.sin(t * 0.011) +
          0.05 * np.random.randn(len(t))).astype("float32")
window = 30
X = np.array([series[i:i+window] for i in range(len(series)-window)])
y = np.array([series[i+window] for i in range(len(series)-window)])
split = int(len(X) * 0.8)
X_train, X_test = X[:split, :, None], X[split:, :, None]
y_train, y_test = y[:split], y[split:]

model = models.Sequential([
    layers.Input((window, 1)),
    layers.LSTM(32),
    layers.Dense(16, activation="relu"),
    layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.1, verbose=2)
loss, mae = model.evaluate(X_test, y_test, verbose=0)
forecast = float(model.predict(X_test[-1:], verbose=0)[0, 0])
print(f"Test MAE: {mae:.4f}")
print(f"Next-step prediction: {forecast:.4f}")
