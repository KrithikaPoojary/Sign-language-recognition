from flask import Flask, render_template, request, redirect, session, jsonify
import sqlite3
import os
import base64
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import io
import mediapipe as mp
import json

app = Flask(__name__)
app.secret_key = "secret123"
db_path = "users.db"

# -----------------------------
# Model + Labels
# -----------------------------
MODEL_PATH = "mobilenet_gesture.h5"
LABELS_JSON = "labels.json"

model, labels = None, {}
try:
    model = load_model(MODEL_PATH)
    with open(LABELS_JSON, "r") as f:
        labels = json.load(f)
    print("✅ Model & labels loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")

IMG_SIZE = 128

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

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
                return redirect('/predict')
            else:
                return "❌ Invalid credentials"
    return render_template('login.html')

@app.route('/predict')
def predict():
    if 'user' not in session:
        return redirect('/login')
    return render_template('predict.html', name=session['user'])

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    if model is None or not labels:
        return jsonify({'error': 'Model or labels not loaded'}), 500

    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

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
        results = hands.process(img_rgb)

        predicted_label, confidence = "No Hand", 0.0

        if results.multi_hand_landmarks:
            h, w, _ = img_np.shape
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]

                xmin, xmax = int(min(x_coords)), int(max(x_coords))
                ymin, ymax = int(min(y_coords)), int(max(y_coords))

                pad = 40
                xmin, xmax = max(0, xmin - pad), min(w, xmax + pad)
                ymin, ymax = max(0, ymin - pad), min(h, ymax + pad)

                hand_img = img_np[ymin:ymax, xmin:xmax]
                if hand_img.size == 0:
                    continue

                hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                hand_img = np.expand_dims(hand_img / 255.0, axis=0)

                preds = model.predict(hand_img, verbose=0)
                class_id = int(np.argmax(preds))
                confidence = float(np.max(preds))

                predicted_label = labels[str(class_id)] if str(class_id) in labels else "Unknown"

        return jsonify({'label': predicted_label, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)
