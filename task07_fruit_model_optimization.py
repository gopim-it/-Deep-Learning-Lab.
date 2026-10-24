"""Task 7: Fruit classification with augmentation, early stopping and LR reduction."""
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path("data/fruits")
if not (ROOT / "train").exists() or not (ROOT / "validation").exists():
    raise SystemExit("Create data/fruits/{train,validation}/<fruit-class>/ image folders; see README.md")

train_ds = tf.keras.utils.image_dataset_from_directory(
    ROOT / "train", image_size=(128, 128), batch_size=32, seed=42)
val_ds = tf.keras.utils.image_dataset_from_directory(
    ROOT / "validation", image_size=(128, 128), batch_size=32, shuffle=False)
classes = train_ds.class_names
print("Fruit classes:", classes)

augment = models.Sequential([
    layers.RandomFlip("horizontal"), layers.RandomRotation(0.08),
    layers.RandomZoom(0.1)
])
model = models.Sequential([
    layers.Input((128, 128, 3)), augment, layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"), layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"), layers.Dropout(0.35),
    layers.Dense(len(classes), activation="softmax")
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"Validation accuracy: {acc:.4f}")
model.save("fruit_classifier.keras")
print("Saved model to fruit_classifier.keras")
