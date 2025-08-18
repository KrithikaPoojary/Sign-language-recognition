import cv2
import os
import time
import mediapipe as mp
import numpy as np

# --- Configuration ---
DATA_DIR = "data"                   # Folder where datasets are stored
TARGET_IMAGE_SIZE = 64              # Size of saved images
TOTAL_IMAGES_PER_GESTURE = 50       # 👈 Only 50 images per gesture
BOUNDING_BOX_MARGIN = 30            # Padding around detected hands
COUNTDOWN_SECONDS = 3               # Countdown before saving starts

# Initialize MediaPipe Hand detector (allow 2 hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,   # detect both hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def collect_gesture_data():
    """Collects images for one gesture (works with one or two hands)."""

    # Ask user for gesture label
    gesture_name = ""
    while not gesture_name:
        gesture_name = input("\nEnter gesture name (e.g., hello, thank_you, namaste, please): ").strip().lower()
        if not gesture_name:
            print("Gesture name cannot be empty. Please try again.")

    gesture_path = os.path.join(DATA_DIR, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)

    # Count existing images
    existing_images = len([f for f in os.listdir(gesture_path) if f.endswith(".jpg")])
    print(f"Found {existing_images} existing images for '{gesture_name}'.")
    print(f"Target: {TOTAL_IMAGES_PER_GESTURE} images.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    count = existing_images
    start_saving = False
    countdown_active = False
    countdown_start_time = 0

    print(f"\n📸 Preparing to collect data for '{gesture_name}' gesture.")
    print("Press 'S' to start saving after countdown, 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(img_rgb)
        segmentation_results = selfie_segmentation.process(img_rgb)

        # Remove background
        condition = np.stack((segmentation_results.segmentation_mask,) * 3, axis=-1) > 0.1
        black_background = np.zeros(frame.shape, dtype=np.uint8)
        segmented_frame = np.where(condition, frame, black_background)

        display_frame = segmented_frame.copy()

        if hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_coords, y_coords = [], []

            # Collect all coords for both hands
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x_coords.extend([lm.x for lm in hand_landmarks.landmark])
                y_coords.extend([lm.y for lm in hand_landmarks.landmark])

            if x_coords and y_coords:
                x_min, y_min = int(min(x_coords) * w), int(min(y_coords) * h)
                x_max, y_max = int(max(x_coords) * w), int(max(y_coords) * h)

                # Add margin
                x_min, y_min = max(x_min - BOUNDING_BOX_MARGIN, 0), max(y_min - BOUNDING_BOX_MARGIN, 0)
                x_max, y_max = min(x_max + BOUNDING_BOX_MARGIN, w), min(y_max + BOUNDING_BOX_MARGIN, h)

                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # Crop both hands together
                hand_img_raw = segmented_frame[y_min:y_max, x_min:x_max]

                if start_saving and hand_img_raw.size > 0:
                    if count < TOTAL_IMAGES_PER_GESTURE:
                        try:
                            resized = cv2.resize(hand_img_raw, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))
                            img_path = os.path.join(gesture_path, f"{count}.jpg")
                            cv2.imwrite(img_path, resized)
                            count += 1
                        except cv2.error as e:
                            print(f"⚠️ Could not save image: {e}")
                    else:
                        start_saving = False
                        print(f"✅ Target {TOTAL_IMAGES_PER_GESTURE} images reached for '{gesture_name}'.")

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
