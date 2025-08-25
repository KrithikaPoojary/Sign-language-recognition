import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
<<<<<<< HEAD

# -------------------------------
# Settings
# -------------------------------
DATA_DIR = "dataset"     # <- folder with subfolders: good, i_love_u, namaste, nice, yes
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
MODEL_OUT = "mobilenet_gesture.h5"
LABELS_JSON = "labels.json"

# -------------------------------
# Data Preparation
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
=======
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Dataset folder
data_dir = "data"
img_size = 128  # MobileNetV2 expects 96+, using 128 for better accuracy

X, y = [], []
labels = sorted(os.listdir(data_dir))
labels = [label for label in labels if os.path.isdir(os.path.join(data_dir, label))]
>>>>>>> 53b77cdd1a8797ac3e1e6ca9261c7552fe43e695

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation"
)

<<<<<<< HEAD
# -------------------------------
# Save labels mapping
# -------------------------------
labels = {v: k for k, v in train_gen.class_indices.items()}  # {0: "good", 1: "i_love_u", ...}
with open(LABELS_JSON, "w") as f:
    json.dump(labels, f)

print("Saved label mapping:", labels)

# -------------------------------
# Model - MobileNetV2
# -------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
=======
print(f"Loading dataset from '{data_dir}' with labels: {labels}...")
total_images_loaded = 0
for label in labels:
    folder_path = os.path.join(data_dir, label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = cv2.resize(img, (img_size, img_size))
        X.append(img)
        y.append(label_dict[label])
        total_images_loaded += 1

if total_images_loaded == 0:
    print("Error: No images were loaded.")
    exit()

X = np.array(X) / 255.0
y = to_categorical(np.array(y), num_classes=len(labels))
>>>>>>> 53b77cdd1a8797ac3e1e6ca9261c7552fe43e695

base_model.trainable = False  # freeze feature extractor

<<<<<<< HEAD
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(train_gen.class_indices), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# -------------------------------
# Training
# -------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# -------------------------------
# Save model
# -------------------------------
model.save(MODEL_OUT)
print(f"✅ Model saved to {MODEL_OUT}")
print(f"✅ Labels saved to {LABELS_JSON}")
=======
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- MobileNetV2 Base Model ---
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # freeze feature extractor

# --- Custom Classifier Head ---
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(labels), activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# --- Data Augmentation ---
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

print("Training model with MobileNetV2...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=15
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save model and labels
model.save("gesture_model.h5")
print("Model saved as gesture_model.h5")

with open("labels.json", "w") as f:
    json.dump(labels, f)
print("Labels saved to labels.json")

# --- Plot training history ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.show()
>>>>>>> 53b77cdd1a8797ac3e1e6ca9261c7552fe43e695
