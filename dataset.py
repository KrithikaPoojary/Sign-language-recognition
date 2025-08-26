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
=======
def collect_gesture_data():
    """Collects images for one gesture and updates labels.json."""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Update labels.json if new gesture
    if gesture_name not in labels:
        labels.append(gesture_name)
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f)
        print(f"✅ Added '{gesture_name}' to {LABELS_FILE}")

    # Count existing images
    existing_images = len([f for f in os.listdir(gesture_path) if f.endswith(".jpg")])
    print(f"Found {existing_images} existing images for '{gesture_name}'.")
    print(f"Target: {TOTAL_IMAGES_PER_GESTURE} images.")

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
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x_coords.extend([lm.x for lm in hand_landmarks.landmark])
                y_coords.extend([lm.y for lm in hand_landmarks.landmark])


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

<<<<<<< HEAD
    cv2.imshow("Dataset Capture", frame)

cap.release()
cv2.destroyAllWindows()
                hand_img_raw = segmented_frame[y_min:y_max, x_min:x_max]

                if start_saving and hand_img_raw.size > 0:
                    if count < TOTAL_IMAGES_PER_GESTURE:
                        try:
                            resized = cv2.resize(hand_img_raw, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))
                            img_path = os.path.join(gesture_path, f"{count}.jpg")
                            cv2.imwrite(img_path, resized)
                            count += 1
                        except cv2.error as e:
                            print(f"Could not save image: {e}")
                    else:
                        start_saving = False
                        print(f"🎯 Target {TOTAL_IMAGES_PER_GESTURE} images reached for '{gesture_name}'.")

        # Overlay text
        status_text = f"Collected: {count}/{TOTAL_IMAGES_PER_GESTURE}"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if not start_saving and not countdown_active:
            cv2.putText(display_frame, "Press 'S' to start saving...", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif countdown_active:
            remaining = int(COUNTDOWN_SECONDS - (time.time() - countdown_start_time))
            if remaining > 0:
                cv2.putText(display_frame, f"Saving in: {remaining}...", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            else:
                countdown_active = False
                start_saving = True
                print("✅ Saving started!")
        elif start_saving:
            cv2.putText(display_frame, "SAVING IMAGES...", (display_frame.shape[1] - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Collecting Gesture Images", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not start_saving and not countdown_active:
            countdown_active = True
            countdown_start_time = time.time()
        elif key == ord('q') or (start_saving and count >= TOTAL_IMAGES_PER_GESTURE):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Collection complete for '{gesture_name}'. Total images: {count}")

if __name__ == "__main__":
    collect_gesture_data()
