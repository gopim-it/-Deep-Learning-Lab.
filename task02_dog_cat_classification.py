"""Task 2: Dog vs cat classification. Requires data/dogs_cats folders; see README."""
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path("data/dogs_cats")
TRAIN, VALID = ROOT / "train", ROOT / "validation"
if not TRAIN.exists() or not VALID.exists():
    raise SystemExit("Create data/dogs_cats/{train,validation}/{cats,dogs}; see README.md")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN, image_size=(128, 128), batch_size=32, seed=42)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID, image_size=(128, 128), batch_size=32, shuffle=False)
print("Class order:", train_ds.class_names)

model = models.Sequential([
    layers.Input((128, 128, 3)), layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"), layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5)
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"Validation accuracy: {acc:.4f}")
