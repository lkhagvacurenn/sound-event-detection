import os
import librosa
import numpy as np

from tensorflow.keras.models import load_model

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_model",
    "sound_classifier.h5"
)

# Change this to any test wav file you want
AUDIO_FILE = os.path.join(
    BASE_DIR,
    "test.wav"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = load_model(MODEL_PATH)

print("Model loaded successfully!")

# =====================================================
# CLASS NAMES
# IMPORTANT:
# Must match train.py order exactly
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
# SETTINGS
# =====================================================

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 1024
TARGET_FRAMES = 258   # ~3 seconds at 22050 Hz / hop 256

# =====================================================
# PREPROCESS FUNCTION
# =====================================================

def preprocess_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Keep first TARGET_FRAMES frames
    mel_db = mel_db[:, :TARGET_FRAMES]

    # Pad if shorter
    if mel_db.shape[1] < TARGET_FRAMES:

        pad_width = TARGET_FRAMES - mel_db.shape[1]

        mel_db = np.pad(
            mel_db,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )

    # Normalize same as train.py
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    X = np.array([mel_db], dtype=np.float32)

    # Add channel dimension: (1, 128, 128, 1)
    X = X[..., np.newaxis]

    return X

# =====================================================
# PREDICT
# =====================================================

X = preprocess_audio(AUDIO_FILE)

print("Input shape:", X.shape)

prediction = model.predict(
    X,
    verbose=0
)

predicted_index = int(np.argmax(prediction[0]))

predicted_class = CLASS_NAMES[predicted_index]

confidence = float(prediction[0][predicted_index])

# =====================================================
# RESULT
# =====================================================

print("\nPrediction Result")
print("-------------------")
print("Audio file:", AUDIO_FILE)
print("Predicted Class:", predicted_class)
print(f"Confidence: {confidence:.4f}")

print("\nAll Predictions")
print("-------------------")

for i, class_name in enumerate(CLASS_NAMES):

    print(f"{class_name}: {prediction[0][i]:.4f}")