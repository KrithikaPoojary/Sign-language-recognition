import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json

# -------------------------------
# Load trained model + labels
# -------------------------------
MODEL_PATH = "mobilenet_gesture.h5"
LABELS_JSON = "labels.json"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels (force string keys for safety)
with open(LABELS_JSON, "r") as f:
    labels = json.load(f)
labels = {str(k): v for k, v in labels.items()}

IMG_SIZE = 128

# -------------------------------
# MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils

# -------------------------------
# Webcam loop
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            xmin, xmax = int(min(x_coords)), int(max(x_coords))
            ymin, ymax = int(min(y_coords)), int(max(y_coords))

            # add padding around hand
            pad = 60
            xmin, xmax = max(0, xmin - pad), min(w, xmax + pad)
            ymin, ymax = max(0, ymin - pad), min(h, ymax + pad)

            # crop hand
            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size == 0 or hand_img.shape[0] < 20 or hand_img.shape[1] < 20:
                continue

            # preprocess for model
            hand_img_resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
            hand_img_norm = np.expand_dims(hand_img_resized / 255.0, axis=0)

            # prediction
            preds = model.predict(hand_img_norm, verbose=0)
            class_id = int(np.argmax(preds))
            label = labels[str(class_id)]
            conf = float(np.max(preds)) * 100

            # draw rectangle + label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.1f}%",
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

            # draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # show cropped hand (debug window)
            cv2.imshow("Cropped Hand", cv2.resize(hand_img_resized, (200, 200)))

    cv2.imshow("Hand Gesture Recognition (MobileNetV2)", frame)

    # ESC key to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
