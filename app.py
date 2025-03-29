from flask import Flask, render_template, request, jsonify
import time
import numpy as np
import tensorflow as tf
import pyrebase
from tensorflow.keras.models import load_model
from tensorflow import keras
import json
import threading
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

######################################
# Firebase Configuration
######################################
firebaseConfig = {
    "apiKey": "AIzaSyCQ_6nxMqQRaGXWjbW8NvIj_6fO6jaUVqc",
    "authDomain": "mhms-ea0f3.firebaseapp.com",
    "databaseURL": "https://mhms-ea0f3-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "mhms-ea0f3",
    "storageBucket": "mhms-ea0f3.appspot.com",
    "messagingSenderId": "979865284947",
    "appId": "1:979865284947:web:929602609ecaacb6be31a6"
}
firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

######################################
# Load Models
######################################
ecg_model = load_model("models/ecg_abnormality_model.keras")
with open("models/model_architecture.json", "r") as json_file:
    model_json = json.load(json_file)
fall_model = keras.models.model_from_json(model_json)
fall_model.load_weights("models/model_weights.bin")  # Ensure .h5 weights are used
fall_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

running = False  # Flag to control execution

def preprocess_ecg_data(data):
    if not data or len(data) != 300:
        return None
    ecg = np.array(data).reshape(1, 300, 1).astype(np.float32)
    return tf.convert_to_tensor(ecg)


def predictECG(ecg_values):
    input_tensor = preprocess_ecg_data(ecg_values)
    if input_tensor is None:
        return None
    prediction = ecg_model.predict(input_tensor)
    prob = prediction[0][0]
    return "Abnormal" if prob > 0.5 else "Normal"

def get_ecg_data():
    data = db.child("ECG_Data").order_by_key().limit_to_last(1).get().val()
    if not data:
        return None
    all_ecg_values = []
    for record in data.values():
        if isinstance(record, dict):  # Ensure it’s a dictionary
            for key, values in record.items():
                if isinstance(values, list):
                    all_ecg_values.extend(values)

    if len(all_ecg_values) >= 600:
        return [all_ecg_values[:300], all_ecg_values[300:600], all_ecg_values]
    else:
        return None


def preprocess_motion_data(firebase_data):
    if not firebase_data or not isinstance(firebase_data, dict):
        return None

    for timestamp, subfolders in firebase_data.items():
        if not isinstance(subfolders, dict):
            continue

        for folder_key, samples in subfolders.items():
            if not isinstance(samples, list) or len(samples) < 390:  # Allow slight variation
                continue

            valid_samples = [s for s in samples if isinstance(s, str)]
            if len(valid_samples) < 390:
                continue

            try:
                # Convert string values to float
                window_data = np.array([list(map(float, s.split(','))) for s in valid_samples])

                # **Pad if Less than 400 Samples**  
                while window_data.shape[0] < 400:
                    window_data = np.vstack([window_data, window_data[-1]])  # Repeat last row

                # Normalize data  
                scaler = StandardScaler()
                window_norm = scaler.fit_transform(window_data)

                return np.expand_dims(window_norm.T, axis=0)  # Ensure correct shape for model

            except Exception as e:
                continue
    return None

def predictFall():
    # Retrieve latest motion data from Firebase
    motion_data = db.child("Motion_Data").order_by_key().limit_to_last(1).get().val()

    if not motion_data:
        return {"fall_detected": False}

    # Preprocess motion data
    batch_data = preprocess_motion_data(motion_data)
    if batch_data is None:
        return {"fall_detected": False}

    # Make a prediction
    prediction = (fall_model.predict(batch_data) > 0.5).astype(int)

    if prediction[0][0] == 1:
        return {"fall_detected": True}
    else:
        return {"fall_detected": False}

def save_results_to_firebase(ecg_condition, fall_result):
    timestamp = int(time.time() * 1000)
    db.child("Predictions").child(timestamp).set({
        "ECG_Condition": ecg_condition,
        "Fall_Detection": fall_result,
        "timestamp": timestamp
    })
    print("✅ ECG and Fall Detection Results Stored in Firebase")


def run_diagnostics():
    global running
    while running:  # ✅ **Added Check Here**
        print("\n🔄 Running ECG and Fall Detection...")

        # Step 1: ECG Prediction
        ecg_data = get_ecg_data()
        if not ecg_data:
            print("⚠️ No valid ECG data for prediction.")
            time.sleep(2)
            continue

        first_prediction = predictECG(ecg_data[0])
        second_prediction = predictECG(ecg_data[1])
        final_ecg_condition = "Abnormal" if first_prediction == "Abnormal" or second_prediction == "Abnormal" else "Normal"

        # Step 2: Fall Detection
        fall_result = predictFall()

        # Step 3: Save both results to Firebase
        save_results_to_firebase(final_ecg_condition, fall_result)

        # Wait for 2 seconds before the next cycle
        time.sleep(2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global running
    if not running:
        running = True
        thread = threading.Thread(target=run_diagnostics)
        thread.start()
    return jsonify({"status": "Started"})

@app.route('/stop', methods=['POST'])
def stop():
    global running
    running = False
    return jsonify({"status": "Stopped"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
