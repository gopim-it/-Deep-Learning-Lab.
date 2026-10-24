"""Task 10: Transfer learning with MobileNetV2 for a folder-based image dataset.
Uses data/fruits/{train,validation}/<class>/ folders; see README.md.
"""
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path("data/fruits")
if not (ROOT / "train").exists() or not (ROOT / "validation").exists():
    raise SystemExit("Prepare data/fruits/{train,validation}/<class>/ folders; see README.md")
train_ds = tf.keras.utils.image_dataset_from_directory(
    ROOT / "train", image_size=(160, 160), batch_size=32)
val_ds = tf.keras.utils.image_dataset_from_directory(
    ROOT / "validation", image_size=(160, 160), batch_size=32, shuffle=False)
num_classes = len(train_ds.class_names)

base = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3), include_top=False, weights="imagenet")
base.trainable = False
inputs = layers.Input((160, 160, 3))
x = layers.Rescaling(1./127.5, offset=-1)(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5)
loss, acc = model.evaluate(val_ds, verbose=0)
print("Class order:", train_ds.class_names)
print(f"Validation accuracy: {acc:.4f}")
model.save("transfer_fruit_classifier.keras")
print("Saved model to transfer_fruit_classifier.keras")
