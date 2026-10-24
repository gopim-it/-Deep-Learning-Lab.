"""Task 4: Autoencoder for short synthetic music-like note sequences.
This is a demonstration of sequence reconstruction, not a trained music composer.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

np.random.seed(7)
tf.random.set_seed(7)
n_samples, seq_len, n_notes = 2000, 32, 12
# Generate simple note patterns with a repeating motif and small variation.
data = np.zeros((n_samples, seq_len, n_notes), dtype="float32")
for i in range(n_samples):
    motif = np.random.randint(0, n_notes, size=4)
    for t in range(seq_len):
        data[i, t, motif[t % 4]] = 1.0
        if np.random.rand() < 0.08:
            data[i, t, np.random.randint(n_notes)] = 1.0

inputs = layers.Input((seq_len, n_notes))
x = layers.Flatten()(inputs)
x = layers.Dense(64, activation="relu")(x)
latent = layers.Dense(16, activation="relu", name="latent")(x)
x = layers.Dense(64, activation="relu")(latent)
x = layers.Dense(seq_len * n_notes, activation="sigmoid")(x)
outputs = layers.Reshape((seq_len, n_notes))(x)
model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(data, data, epochs=8, batch_size=32, validation_split=0.1, verbose=2)
reconstructed = model.predict(data[:1], verbose=0)[0]
np.save("reconstructed_music_sequence.npy", reconstructed)
print("Saved reconstructed sequence to reconstructed_music_sequence.npy")
print("Sequence shape:", reconstructed.shape)
