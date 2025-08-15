import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import os

# Load trained model
model = load_model('gesture_model.h5')

# Manually define class labels (in order of training)
labels = ['hello', 'thank_you', 'namaste']  # Add more if needed

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
print("📷 Starting webcam for real-time prediction...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip frame horizontally and convert color
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            x1 = max(0, int(x_min * w) - 20)
            y1 = max(0, int(y_min * h) - 20)
            x2 = min(w, int(x_max * w) + 20)
            y2 = min(h, int(y_max * h) + 20)

            # Crop and preprocess the hand image
            hand_img = frame[y1:y2, x1:x2]
            if hand_img.size == 0:
                continue

            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            # Predict gesture
            prediction = model.predict(hand_img)
            pred_class = np.argmax(prediction)
            pred_label = labels[pred_class]
            confidence = np.max(prediction)

            print(f"Predicted: {pred_label}, Confidence: {confidence:.2f}")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Show label only if confidence > threshold
            if confidence > 0.7:
                cv2.putText(frame, f"{pred_label} ({confidence*100:.1f}%)",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not Confident",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display frame
    cv2.imshow("Sign Language Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
