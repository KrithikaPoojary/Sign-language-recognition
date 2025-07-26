import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt # For plotting training history

# Dataset folder
data_dir = "data"
img_size = 64 # This must match TARGET_IMAGE_SIZE in dataset.py and app.py

X, y = [], []
labels = sorted(os.listdir(data_dir))
# Filter out non-directory entries and ensure labels are consistent
labels = [label for label in labels if os.path.isdir(os.path.join(data_dir, label))]

if not labels:
    print(f"Error: No gesture data found in '{data_dir}'. Please run dataset.py first.")
    exit()

label_dict = {label: idx for idx, label in enumerate(labels)}

print(f"Loading dataset from '{data_dir}' with labels: {labels}...")
total_images_loaded = 0
for label in labels:
    folder_path = os.path.join(data_dir, label)
    if not os.path.isdir(folder_path):
        print(f"Skipping {folder_path} as it's not a directory.")
        continue
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping.")
            continue
        # Ensure image is resized to match model input
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = cv2.resize(img, (img_size, img_size))
        
        X.append(img)
        y.append(label_dict[label])
        total_images_loaded += 1

if total_images_loaded == 0:
    print("Error: No images were loaded. Please ensure dataset.py collected images correctly.")
    exit()

X = np.array(X) / 255.0 # Normalize pixel values
y = to_categorical(np.array(y), num_classes=len(labels))

print(f"Dataset loaded: {len(X)} images, {len(labels)} classes.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratify for balanced classes

# Model Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Regularization to prevent overfitting
    Dense(len(labels), activation='softmax') # Output layer matches number of classes
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
history = model.fit(X_train, y_train, epochs=20, # Increased epochs for better training
                    validation_data=(X_test, y_test), batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show() # Display the plots

# Save model
model.save("gesture_model.h5")
print("✅ Model saved as gesture_model.h5")