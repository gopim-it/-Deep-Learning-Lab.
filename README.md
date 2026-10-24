# Deep Learning Lab — 12 Tasks

Each task is a separate Python file. The programs use TensorFlow/Keras, except the YOLO task, which uses Ultralytics.

## Tasks
1. `task01_mnist_classification.py` — MNIST handwritten-digit classification
2. `task02_dog_cat_classification.py` — Dog vs Cat image classification
3. `task03_synthetic_image_generation.py` — Generate synthetic images with a simple GAN
4. `task04_music_autoencoder.py` — Train an autoencoder on synthetic music-like sequences and generate a sequence
5. `task05_time_series_forecasting.py` — Time-series forecasting with an LSTM
6. `task06_car_detection_yolo.py` — Detect cars in an image using YOLO
7. `task07_fruit_model_optimization.py` — Fruit image classification with data augmentation and callbacks
8. `task08_dense_binary_classification.py` — Binary classification with a dense neural network
9. `task09_cnn_image_classification.py` — CNN classification on CIFAR-10
10. `task10_transfer_learning.py` — Transfer learning with MobileNetV2
11. `task11_text_sentiment_rnn.py` — Text sentiment classification with an Embedding + LSTM model
12. `task12_denoising_autoencoder.py` — Denoising autoencoder on MNIST

## Requirements

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Some datasets are downloaded automatically by Keras on first run and require internet access.
For image-folder tasks, download/prepare the data yourself as described below.

## Run a task

```bash
python task01_mnist_classification.py
```

Run commands from this folder.

## Image-folder datasets

### Task 2 — Dogs and cats
Create this structure and place images in the folders:

```text
data/dogs_cats/
  train/
    cats/
    dogs/
  validation/
    cats/
    dogs/
```

### Task 7 — Fruits
Use this structure (class names are folder names):

```text
data/fruits/
  train/
    apple/
    banana/
    orange/
  validation/
    apple/
    banana/
    orange/
```

### Task 10 — Transfer learning
Uses the same fruit-folder structure as Task 7. Pretrained MobileNetV2 weights are downloaded automatically.

### Task 6 — YOLO car detection
Install `ultralytics` using requirements. Put an image at `data/car.jpg`, or pass an image path as the first command-line argument. The default YOLO model is downloaded automatically on first run. It detects the COCO `car` class.

## Notes
- Accuracy and predicted classes vary with training, random initialization, and the dataset.
- Never present sample output as a guaranteed result. Each script prints its actual output when executed.
- Large datasets and model weights are not included in this repository.
