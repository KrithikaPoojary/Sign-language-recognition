from flask import Flask, render_template, request, redirect, session, jsonify
import sqlite3
import os
import json
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
from PIL import Image
import io
import mediapipe as mp

app = Flask(__name__)
app.secret_key = "secret123"  # change to env variable in production
db_path = "users.db"

# -----------------------------
# Model & Labels
# -----------------------------
MODEL_PATH = "mobilenet_gesture.h5"
LABELS_PATH = "labels.json"

model = None
labels = {}

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        model = load_model(MODEL_PATH)

        with open(LABELS_PATH, "r") as f:
            labels = json.load(f)   # dict { "0": "Good", "1": "Hello", ... }

        print("✅ Model and labels loaded successfully")
    else:
        print("❌ Model or labels file missing")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# -----------------------------
# Database Init
# -----------------------------
def init_db():
    with sqlite3.connect(db_path) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users
                        (id INTEGER PRIMARY KEY,
                         name TEXT,
                         email TEXT UNIQUE,
                         password TEXT)''')
    print("✅ Database initialized")

init_db()

# -----------------------------
# Routes
# -----------------------------
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
                return "❌ User with this email already exists."
            except Exception as e:
                return f"❌ Error during signup: {e}"
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
                return "❌ Invalid credentials"
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

    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_np = np.array(img_pil)

        if img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # Detect hand with MediaPipe
        with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as hands_processor:
            hand_results = hands_processor.process(img_rgb)

        predicted_label = "No Hand Detected"
        confidence = 0.0

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                h, w, _ = img_np.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min, y_min = int(min(x_coords) * w), int(min(y_coords) * h)
                x_max, y_max = int(max(x_coords) * w), int(max(y_coords) * h)

                margin = 30
                x_min, y_min = max(x_min - margin, 0), max(y_min - margin, 0)
                x_max, y_max = min(x_max + margin, w), min(y_max + margin, h)

                hand_img = img_np[y_min:y_max, x_min:x_max]

                if hand_img.size > 0:
                    TARGET_SIZE = 128  # match training size
                    img_resized = cv2.resize(hand_img, (TARGET_SIZE, TARGET_SIZE))
                    img_normalized = img_resized / 255.0
                    img_input = np.expand_dims(img_normalized, axis=0)

                    prediction = model.predict(img_input, verbose=0)
                    predicted_index = int(np.argmax(prediction))
                    confidence = float(np.max(prediction))

                    predicted_label = labels.get(str(predicted_index), "Unknown")

        return jsonify({'label': predicted_label, 'confidence': confidence})

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
