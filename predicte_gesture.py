import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
from collections import deque, Counter

# -------------------------------
# Load model + labels
# -------------------------------
MODEL_PATH = "mobilenet_gesture.h5"
LABELS_JSON = "labels.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_JSON, "r") as f:
    labels = json.load(f)

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
# Prediction smoothing
# -------------------------------
PRED_HISTORY = 15  # number of frames to keep
pred_queue = deque(maxlen=PRED_HISTORY)

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

            pad = 40
            xmin, xmax = max(0, xmin - pad), min(w, xmax + pad)
            ymin, ymax = max(0, ymin - pad), min(h, ymax + pad)

            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size == 0:
                continue

            hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
            hand_img = hand_img.astype("float32") / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            preds = model.predict(hand_img, verbose=0)
            class_id = int(np.argmax(preds))
            label = labels[str(class_id)]
            conf = float(np.max(preds)) * 100

            # Save prediction into history
            pred_queue.append(label)

            # Use majority vote for stable output
            if len(pred_queue) == PRED_HISTORY:
                most_common = Counter(pred_queue).most_common(1)[0][0]
            else:
                most_common = label

            # Draw results
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{most_common} ({conf:.1f}%)",
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Stable Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
