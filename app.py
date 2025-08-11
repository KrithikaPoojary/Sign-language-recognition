from flask import Flask, render_template, request, redirect, session, jsonify
import sqlite3
import os
import subprocess
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
from PIL import Image
import io
import mediapipe as mp

app = Flask(__name__)
app.secret_key = "secret123" # In a real app, use a strong, environment variable for this
db_path = "users.db"
model_path = "gesture_model.h5"
data_dir = "data" # Directory where gesture data is stored and labels are derived

# --- Auto-train if model doesn't exist ---
if not os.path.exists(model_path):
    print("gesture_model.h5 not found. Training model now...")
    result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"Error training model: {result.stderr}")
        print(f"Stdout: {result.stdout}")
    else:
        print("Model trained and saved!")
        print(f"Stdout: {result.stdout}")

# Load the trained model and labels
model = None
labels = []
try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        if os.path.exists(data_dir):
            labels = sorted(os.listdir(data_dir))
            labels = [label for label in labels if os.path.isdir(os.path.join(data_dir, label))]
        print(f"Loaded model with labels: {labels}")
    else:
        print("Model file not found after attempted training.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# DB Initialization
def init_db():
    with sqlite3.connect(db_path) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users
                        (id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, password TEXT)''')
    print("Database initialized.")

init_db()

@app.route('/')
def index():
    return redirect('/login')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        with sqlite3.connect(db_path) as conn:
            try:
                conn.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                             (name, email, password))
                conn.commit()
                return redirect('/login')
            except sqlite3.IntegrityError:
                return "User with this email already exists."
            except Exception as e:
                return f"An error occurred during signup: {e}"
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM users WHERE email=? AND password=?", (email, password))
            row = cur.fetchone()
            if row:
                session['user'] = row[0]
                return redirect('/home')
            else:
                return "Invalid credentials"
    return render_template('login.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect('/login')
    return render_template('home.html', name=session['user'])

@app.route('/predict')
def predict():
    if 'user' not in session:
        return redirect('/login')
    return render_template('predict.html', name=session['user'])

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    if model is None or not labels:
        return jsonify({'error': 'Model or labels not loaded on server'}), 500

    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = data['image'].split(',')[1]
    try:
        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_np = np.array(img_pil)

        if img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_rgb_mp = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # --- Instantiate processors for each request ---
        # This is the key change to avoid timestamp issues
        with mp.solutions.hands.Hands(
            static_image_mode=True, # Important for processing discrete images
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as hands_processor:
            hand_results = hands_processor.process(img_rgb_mp)

        with mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        ) as selfie_segmentation_processor:
            segmentation_results = selfie_segmentation_processor.process(img_rgb_mp)
        # --- End of key change ---

        condition = np.stack((segmentation_results.segmentation_mask,) * 3, axis=-1) > 0.1
        black_background = np.zeros(img_np.shape, dtype=np.uint8)
        segmented_frame = np.where(condition, img_np, black_background)

        predicted_label = "No Hand Detected"
        confidence = 0.0

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                h, w, _ = img_np.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min, y_min = int(min(x_coords) * w), int(min(y_coords) * h)
                x_max, y_max = int(max(x_coords) * w), int(max(y_coords) * h)

                BOUNDING_BOX_MARGIN = 30
                x_min, y_min = max(x_min - BOUNDING_BOX_MARGIN, 0), max(y_min - BOUNDING_BOX_MARGIN, 0)
                x_max, y_max = min(x_max + BOUNDING_BOX_MARGIN, w), min(y_max + BOUNDING_BOX_MARGIN, h)

                hand_img_cropped = segmented_frame[y_min:y_max, x_min:x_max]

                if hand_img_cropped.size > 0 and hand_img_cropped.shape[0] > 0 and hand_img_cropped.shape[1] > 0:
                    TARGET_IMAGE_SIZE = 64
                    img_resized = cv2.resize(hand_img_cropped, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))
                    img_normalized = img_resized / 255.0
                    img_input = np.expand_dims(img_normalized, axis=0)

                    prediction = model.predict(img_input, verbose=0) # Added verbose=0 to silence prediction output
                    predicted_class_index = np.argmax(prediction)
                    confidence = float(np.max(prediction))

                    if 0 <= predicted_class_index < len(labels):
                        predicted_label = labels[predicted_class_index]
                    else:
                        predicted_label = "Unknown Label"
                else:
                    predicted_label = "Hand Cropping Failed"
        
        return jsonify({'label': predicted_label, 'confidence': confidence})

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)