import os
import librosa
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# =========================================================
# CLASS NAMES
# Folder names must match these exactly:
# dataset/background
# dataset/dog
# dataset/door
# dataset/speech
# dataset/walk
# dataset/water
# =========================================================

CLASS_NAMES = [
    "background",
    "dog",
    "door",
    "speech",
    "walk",
    "water"
]

CLASS_MAP = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

NUM_CLASSES = len(CLASS_NAMES)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(__file__)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset"
)

SAVE_FOLDER = os.path.join(
    BASE_DIR,
    "saved_model"
)

os.makedirs(SAVE_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(
    SAVE_FOLDER,
    "sound_classifier.h5"
)

# =========================================================
# SETTINGS
# =========================================================

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 1024
TARGET_FRAMES = 258   # ~3 seconds at 22050 Hz / hop 256

# =========================================================
# DATA ARRAYS
# =========================================================

X = []
y = []

# =========================================================
# PREPROCESS FUNCTION
# =========================================================

def create_mel_spectrogram(file_path, augment=False):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Add light Gaussian noise so the model learns to handle
    # real microphone recordings (not just clean dataset audio)
    if augment:
        noise = np.random.normal(0, 0.005, audio.shape)
        audio = audio + noise

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

    # Keep first TARGET_FRAMES time frames
    mel_db = mel_db[:, :TARGET_FRAMES]

    # Pad if shorter than TARGET_FRAMES
    if mel_db.shape[1] < TARGET_FRAMES:

        pad_width = TARGET_FRAMES - mel_db.shape[1]

        mel_db = np.pad(
            mel_db,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )

    # Normalize 0 to 1
    mel_db = (mel_db - mel_db.min()) / (
        mel_db.max() - mel_db.min() + 1e-6
    )

    return mel_db

# =========================================================
# COUNT FILES
# =========================================================

total_files = 0

for class_name in CLASS_NAMES:

    class_folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_folder):

        print(f"WARNING: Folder not found: {class_folder}")
        continue

    wav_files = [
        file for file in os.listdir(class_folder)
        if file.lower().endswith(".wav")
    ]

    total_files += len(wav_files)

print(f"Total WAV files found: {total_files}")

# =========================================================
# LOAD DATASET FROM FOLDERS
# =========================================================

processed_count = 0

for class_name in CLASS_NAMES:

    class_folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_folder):
        continue

    label = CLASS_MAP[class_name]

    wav_files = [
        file for file in os.listdir(class_folder)
        if file.lower().endswith(".wav")
    ]

    print(f"\nLoading class: {class_name} ({len(wav_files)} files)")

    for file in wav_files:

        processed_count += 1

        file_path = os.path.join(
            class_folder,
            file
        )

        print(
            f"Processing {processed_count}/{total_files}: "
            f"{class_name}/{file}"
        )

        try:

            # Original sample
            mel_db = create_mel_spectrogram(file_path, augment=False)
            X.append(mel_db)
            y.append(label)

            # Augmented copy with added noise — doubles training data
            mel_db_aug = create_mel_spectrogram(file_path, augment=True)
            X.append(mel_db_aug)
            y.append(label)

        except Exception as e:

            print("ERROR:", file_path, e)

# =========================================================
# CONVERT TO NUMPY
# =========================================================

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

if len(X) == 0:
    raise Exception(
        "No audio files loaded. Check dataset folder path and WAV files."
    )

# =========================================================
# PREPARE FOR CNN
# =========================================================

# Add channel dimension: (samples, 128, 128, 1)
X = X[..., np.newaxis]

# Split using integer labels first
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert labels to one-hot after splitting
y_train = to_categorical(
    y_train_int,
    num_classes=NUM_CLASSES
)

y_test = to_categorical(
    y_test_int,
    num_classes=NUM_CLASSES
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTrain class counts:")
for i, class_name in enumerate(CLASS_NAMES):
    print(class_name, np.sum(y_train_int == i))

print("\nTest class counts:")
for i, class_name in enumerate(CLASS_NAMES):
    print(class_name, np.sum(y_test_int == i))

# =========================================================
# BUILD CNN MODEL
# Stable model for small/noisy dataset
# =========================================================

model = Sequential()

model.add(
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, TARGET_FRAMES, 1))
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(NUM_CLASSES, activation="softmax"))

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================================================
# MODEL SUMMARY
# =========================================================

model.summary()

# =========================================================
# CALLBACKS
# =========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =========================================================
# TRAIN MODEL
# =========================================================

history = model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[
        early_stop,
        checkpoint
    ]
)

# =========================================================
# EVALUATE MODEL
# =========================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# =========================================================
# SAVE FINAL MODEL
# =========================================================

model.save(MODEL_PATH)

print(f"\nModel saved successfully at: {MODEL_PATH}")