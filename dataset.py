import cv2
import os
import time
import mediapipe as mp
import numpy as np

# --- Configuration ---
DATA_DIR = "data"                   # Main directory to save datasets
TARGET_IMAGE_SIZE = 64              # Resized dimensions for saved images (must match train_model.py and app.py)
TOTAL_IMAGES_PER_GESTURE = 250      # Number of images to collect per gesture
BOUNDING_BOX_MARGIN = 30            # Padding around the detected hand
COUNTDOWN_SECONDS = 3               # Countdown before image capture begins

# Initialize MediaPipe Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1) # model_selection 0 for general, 1 for landscape

def collect_gesture_data():
    """Collects image data for a specified hand gesture with background removal."""

    # Ask user for gesture label
    gesture_name = ""
    while not gesture_name:
        gesture_name = input("\nEnter gesture name (e.g., hello, thank_you, stop): ").strip().lower()
        if not gesture_name:
            print("Gesture name cannot be empty. Please try again.")

    gesture_path = os.path.join(DATA_DIR, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)

    # Check existing images to avoid overwriting and continue collection
    existing_images = 0
    if os.path.exists(gesture_path):
        existing_images = len([name for name in os.listdir(gesture_path) if name.endswith(".jpg")]) # Count only .jpg files

    print(f"Found {existing_images} existing images for '{gesture_name}'. Continuing collection...")
    print(f"Targeting {TOTAL_IMAGES_PER_GESTURE} images for this gesture.")

    # Capture from webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check if it's connected and not in use.")
        return

    count = existing_images # Start count from existing images
    start_saving = False
    countdown_active = False
    countdown_start_time = 0

    print(f"\n📸 Preparing to collect data for '{gesture_name}' gesture.")
    print(f"Instructions: Position your hand clearly in the frame, ideally against a plain background for best results.")
    print(f"Press 'S' to start saving images after the countdown.")
    print(f"Press 'Q' to quit at any time.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        frame = cv2.flip(frame, 1) # Flip frame horizontally for mirror effect
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process for hand landmarks
        hand_results = hands.process(img_rgb)

        # Process for segmentation mask
        segmentation_results = selfie_segmentation.process(img_rgb)
        
        # Get binary mask for the person (1.0 for person, 0.0 for background)
        # We threshold it to create a clear binary mask. Adjust 0.1 if segmentation is poor.
        condition = np.stack((segmentation_results.segmentation_mask,) * 3, axis=-1) > 0.1
        
        # Create a blank background (black) or use the original background
        black_background = np.zeros(frame.shape, dtype=np.uint8)
        
        # Apply the mask: keep foreground, set background to black
        segmented_frame = np.where(condition, frame, black_background)

        display_frame = segmented_frame.copy() # Use the segmented frame for display and cropping

        hand_detected = False
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_detected = True
                mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Green landmarks
                                       mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)) # Red connections

                # Bounding box around hand
                h, w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min, y_min = int(min(x_coords) * w), int(min(y_coords) * h)
                x_max, y_max = int(max(x_coords) * w), int(max(y_coords) * h)

                # Add padding to box
                x_min, y_min = max(x_min - BOUNDING_BOX_MARGIN, 0), max(y_min - BOUNDING_BOX_MARGIN, 0)
                x_max, y_max = min(x_max + BOUNDING_BOX_MARGIN, w), min(y_max + BOUNDING_BOX_MARGIN, h)

                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2) # Blue bounding box

                # Crop from the segmented frame, not the original frame
                hand_img_raw = segmented_frame[y_min:y_max, x_min:x_max]

                if start_saving and hand_img_raw.size > 0:
                    if count < TOTAL_IMAGES_PER_GESTURE:
                        try:
                            # Ensure the cropped image is not empty before resizing
                            if hand_img_raw.shape[0] > 0 and hand_img_raw.shape[1] > 0:
                                resized = cv2.resize(hand_img_raw, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))
                                img_path = os.path.join(gesture_path, f"{count}.jpg")
                                cv2.imwrite(img_path, resized)
                                count += 1
                        except cv2.error as e:
                            print(f"Warning: Could not resize or save image. {e}")
                    else:
                        start_saving = False # Stop saving once target is reached
                        print(f"✅ Target {TOTAL_IMAGES_PER_GESTURE} images reached for '{gesture_name}'.")

        # Display status and instructions
        status_text = f"Collected: {count}/{TOTAL_IMAGES_PER_GESTURE}"
        cv2.putText(display_frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if not start_saving and not countdown_active:
            cv2.putText(display_frame, "Press 'S' to start saving...", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif countdown_active:
            remaining_time = int(COUNTDOWN_SECONDS - (time.time() - countdown_start_time))
            if remaining_time > 0:
                cv2.putText(display_frame, f"Saving in: {remaining_time}...", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            else:
                countdown_active = False
                start_saving = True
                print("✅ Saving started!")
        elif start_saving:
            cv2.putText(display_frame, "SAVING IMAGES...", (display_frame.shape[1] - 250, 30), # Adjusted position
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        cv2.imshow("Collecting Gesture Images (Background Removed)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not start_saving and not countdown_active:
            countdown_active = True
            countdown_start_time = time.time()
            print(f"Countdown started for {COUNTDOWN_SECONDS} seconds...")
        elif key == ord('q') or (start_saving and count >= TOTAL_IMAGES_PER_GESTURE): # Added condition to stop only if saving
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Data collection session complete for gesture '{gesture_name}'. Total images: {count}")

if __name__ == "__main__":
    collect_gesture_data()