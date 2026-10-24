"""Task 3: A small GAN that generates 28x28 grayscale images from MNIST."""
import tensorflow as tf
from tensorflow.keras import layers, models

tf.random.set_seed(7)
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = x_train[..., None]
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(128, drop_remainder=True)

noise_dim = 100
generator = models.Sequential([
    layers.Input((noise_dim,)), layers.Dense(7*7*128, use_bias=False),
    layers.BatchNormalization(), layers.LeakyReLU(),
    layers.Reshape((7, 7, 128)),
    layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False),
    layers.BatchNormalization(), layers.LeakyReLU(),
    layers.Conv2DTranspose(1, 4, strides=2, padding="same", activation="tanh")
])
discriminator = models.Sequential([
    layers.Input((28, 28, 1)), layers.Conv2D(64, 4, strides=2, padding="same"),
    layers.LeakyReLU(), layers.Dropout(0.3),
    layers.Conv2D(128, 4, strides=2, padding="same"),
    layers.LeakyReLU(), layers.Dropout(0.3), layers.Flatten(),
    layers.Dense(1)
])
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_opt = tf.keras.optimizers.Adam(1e-4)
d_opt = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(real):
    noise = tf.random.normal([tf.shape(real)[0], noise_dim])
    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake = generator(noise, training=True)
        real_logits = discriminator(real, training=True)
        fake_logits = discriminator(fake, training=True)
        g_loss = bce(tf.ones_like(fake_logits), fake_logits)
        d_loss = bce(tf.ones_like(real_logits), real_logits) + bce(tf.zeros_like(fake_logits), fake_logits)
    g_grads = gt.gradient(g_loss, generator.trainable_variables)
    d_grads = dt.gradient(d_loss, discriminator.trainable_variables)
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    return g_loss, d_loss

for epoch in range(3):
    gl, dl = [], []
    for batch in dataset.take(150):
        g_loss, d_loss = train_step(batch)
        gl.append(g_loss); dl.append(d_loss)
    print(f"Epoch {epoch+1}: generator loss={tf.reduce_mean(gl):.4f}, discriminator loss={tf.reduce_mean(dl):.4f}")

noise = tf.random.normal([1, noise_dim])
generated = generator(noise, training=False)[0]
image = tf.cast((generated + 1) * 127.5, tf.uint8)
tf.keras.utils.save_img("generated_image.png", image)
print("Saved one generated image to generated_image.png")
