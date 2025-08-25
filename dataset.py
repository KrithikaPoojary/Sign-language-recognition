import cv2
import os
import mediapipe as mp

# -------------------------------
# Settings
# -------------------------------
DATA_DIR = "dataset"
GESTURES = ["good", "i_love_u", "namaste", "nice", "yes"]
IMG_SIZE = 128
MAX_IMAGES = 300   # per gesture

# -------------------------------
# Create gesture folders
# -------------------------------
for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_DIR, gesture), exist_ok=True)

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
# Open Camera
# -------------------------------
cap = cv2.VideoCapture(0)
current_gesture = None
img_count = 0

print("Press 1-5 to select gesture:")
print("1 = good, 2 = i_love_u, 3 = namaste, 4 = nice, 5 = yes")
print("Press 'q' to stop saving and quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    # Select gesture key
    if key == ord('1'):
        current_gesture = "good"; img_count = 0
        print("➡️ Capturing GOOD")
    elif key == ord('2'):
        current_gesture = "i_love_u"; img_count = 0
        print("➡️ Capturing I LOVE U")
    elif key == ord('3'):
        current_gesture = "namaste"; img_count = 0
        print("➡️ Capturing NAMASTE")
    elif key == ord('4'):
        current_gesture = "nice"; img_count = 0
        print("➡️ Capturing NICE")
    elif key == ord('5'):
        current_gesture = "yes"; img_count = 0
        print("➡️ Capturing YES")
    elif key == ord('q'):
        print("🛑 Stopped.")
        break

    if results.multi_hand_landmarks and current_gesture:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            xmin, xmax = int(min(x_coords)), int(max(x_coords))
            ymin, ymax = int(min(y_coords)), int(max(y_coords))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            if img_count < MAX_IMAGES:
                hand_img = frame[ymin:ymax, xmin:xmax]
                if hand_img.size != 0:
                    hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                    save_path = os.path.join(DATA_DIR, current_gesture, f"{img_count}.jpg")
                    cv2.imwrite(save_path, hand_img)
                    img_count += 1

            cv2.putText(frame, f"{current_gesture}: {img_count}/{MAX_IMAGES}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Dataset Capture", frame)

cap.release()
cv2.destroyAllWindows()
