import cv2
import mediapipe as mp
import numpy as np
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

cap = cv2.VideoCapture(0)
print("📷 Starting webcam for real-time prediction...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

            x1 = max(0, int(x_min * w) - 20)
            y1 = max(0, int(y_min * h) - 20)
            x2 = min(w, int(x_max * w) + 20)
            y2 = min(h, int(y_max * h) + 20)

            hand_img = segmented_frame[y1:y2, x1:x2]
            if hand_img.size == 0:
                continue

            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

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
        break

cap.release()
cv2.destroyAllWindows()
