import os
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

if not labels:
    print(f"Error: No gesture data found in '{data_dir}'. Please run dataset.py first.")
    exit()

label_dict = {label: idx for idx, label in enumerate(labels)}

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

print(f"Dataset loaded: {len(X)} images, {len(labels)} classes.")

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
