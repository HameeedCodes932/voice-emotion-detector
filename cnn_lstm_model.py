import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# 🔸 Emotions Map
emotions_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# 🔸 Labels You Want to Use
observed_emotions = ["angry", "happy", "sad", "neutral"]

# 🔸 Dataset Path
DATA_PATH = "Audio_Speech_Actors_01-24"

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return mfccs.T
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return None

features = []
labels = []
errors = 0
total_files = 0

# 🔸 Loop through dataset
for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            total_files += 1
            file_path = os.path.join(root, file)
            emotion_code = file.split("-")[2]
            emotion = emotions_map.get(emotion_code)
            if emotion not in observed_emotions:
                continue
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(observed_emotions.index(emotion))
            else:
                errors += 1

print(f"✅ Total files processed: {total_files}")
print(f"✅ Features extracted: {len(features)}")
print(f"❌ Skipped due to errors: {errors}")

# 🔸 Padding sequences to same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

X = pad_sequences(features, padding='post')
X = X.reshape(X.shape[0], X.shape[1], 40)  # (samples, time steps, features)

# 🔸 Labels to categorical
y = to_categorical(labels)

# 🔸 Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔸 Build CNN-LSTM model
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 40)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 🔸 Train model
model.fit(x_train, y_train, epochs=40, batch_size=32, validation_data=(x_test, y_test))

# 🔸 Save model
model.save("emotion_model.h5")
print("✅ Model saved as emotion_model.h5")
