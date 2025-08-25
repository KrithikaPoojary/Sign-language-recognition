import cv2
import numpy as np
<<<<<<< HEAD
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
=======
import json
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model('gesture_model.h5')
with open("labels.json", "r") as f:
    labels = json.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Optional: Background segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

>>>>>>> 53b77cdd1a8797ac3e1e6ca9261c7552fe43e695
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

<<<<<<< HEAD
=======
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
>>>>>>> 53b77cdd1a8797ac3e1e6ca9261c7552fe43e695
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

<<<<<<< HEAD
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            xmin, xmax = int(min(x_coords)), int(max(x_coords))
            ymin, ymax = int(min(y_coords)), int(max(y_coords))
=======
    # Remove background
    segmentation_results = selfie_segmentation.process(rgb)
    condition = np.stack((segmentation_results.segmentation_mask,) * 3, axis=-1) > 0.1
    black_bg = np.zeros(frame.shape, dtype=np.uint8)
    segmented_frame = np.where(condition, frame, black_bg)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])
>>>>>>> 53b77cdd1a8797ac3e1e6ca9261c7552fe43e695

            # add padding around hand
            pad = 60
            xmin, xmax = max(0, xmin - pad), min(w, xmax + pad)
            ymin, ymax = max(0, ymin - pad), min(h, ymax + pad)

<<<<<<< HEAD
            # crop hand
            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size == 0 or hand_img.shape[0] < 20 or hand_img.shape[1] < 20:
=======
            hand_img = segmented_frame[y1:y2, x1:x2]
            if hand_img.size == 0:
>>>>>>> 53b77cdd1a8797ac3e1e6ca9261c7552fe43e695
                continue

            # preprocess for model
            hand_img_resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
            hand_img_norm = np.expand_dims(hand_img_resized / 255.0, axis=0)

<<<<<<< HEAD
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
=======
            prediction = model.predict(hand_img, verbose=0)
            pred_class = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > 0.7:
                pred_label = labels[pred_class]
                text = f"{pred_label} ({confidence*100:.1f}%)"
                color = (0, 255, 0)
            else:
                text = "Not confident"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
>>>>>>> 53b77cdd1a8797ac3e1e6ca9261c7552fe43e695
        break

cap.release()
cv2.destroyAllWindows()
