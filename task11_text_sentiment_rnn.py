"""Task 11: Sentiment classification using a small built-in movie-review dataset."""
import tensorflow as tf
from tensorflow.keras import layers, models

max_features, seq_len = 10000, 200
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)
x_train = tf.keras.utils.pad_sequences(x_train, maxlen=seq_len)
x_test = tf.keras.utils.pad_sequences(x_test, maxlen=seq_len)

model = models.Sequential([
    layers.Input((seq_len,)), layers.Embedding(max_features, 64),
    layers.LSTM(32), layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=2)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")
print("Labels: 0 = negative review, 1 = positive review")
