import os
import librosa
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

from tensorflow.keras.models import load_model

# =====================================================
# APP SETUP
# =====================================================

app = Flask(__name__)
CORS(app)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_model",
    "sound_classifier.h5"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================

model = load_model(MODEL_PATH)

print("Model loaded successfully!")

# =====================================================
# CLASS NAMES
# IMPORTANT:
# This order MUST match your training code label order.
# =====================================================

CLASS_NAMES = [
    "background",
    "dog",
    "door",
    "speech",
    "walk",
    "water"
]

# =====================================================
# PREPROCESS FUNCTION
# Same preprocessing as training code
# =====================================================

def preprocess_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=1024
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Keep first 128 time frames
    mel_db = mel_db[:, :128]

    # Pad if audio is shorter
    if mel_db.shape[1] < 128:

        pad_width = 128 - mel_db.shape[1]

        mel_db = np.pad(
            mel_db,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )

    # Normalize values
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    X = np.array([mel_db])

    # Add channel dimension: (1, 128, 128, 1)
    X = X[..., np.newaxis]

    return X

# =====================================================
# PREDICT ROUTE
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:

        return jsonify({
            "error": "No audio file uploaded"
        }), 400

    audio_file = request.files["audio"]

    file_path = os.path.join(
        UPLOAD_FOLDER,
        audio_file.filename
    )

    audio_file.save(file_path)

    try:

        X = preprocess_audio(file_path)

        prediction = model.predict(
            X,
            verbose=0
        )

        predicted_index = int(np.argmax(prediction[0]))

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(prediction[0][predicted_index])

        return jsonify({
            "class": predicted_class,
            "confidence": confidence,
            "all_predictions": {
                CLASS_NAMES[i]: float(prediction[0][i])
                for i in range(len(CLASS_NAMES))
            }
        })

    except Exception as e:

        import traceback
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500

# =====================================================
# START SERVER
# =====================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )