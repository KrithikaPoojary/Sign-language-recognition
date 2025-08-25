import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

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

base_model.trainable = False  # freeze feature extractor

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
