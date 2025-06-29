import os
import librosa
import numpy as np
import soundfile

# Dataset folder ka path — yahan apna path lagao
dataset_path = dataset_path = "C:/Users/fazli/voice_emotion_dataset/Audio_Speech_Actors_01-24_16k/"


# Emotions map
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Audio se features nikaalne wali function
def extract_features(file_path):
    with soundfile.SoundFile(file_path) as sound_file:
        audio = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
        return np.mean(mfccs.T, axis=0)

# Sab files per loop
features = []
labels = []

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                try:
                    mfcc = extract_features(file_path)
                    emotion = emotion_map[file.split("-")[2]]
                    features.append(mfcc)
                    labels.append(emotion)
                except Exception as e:
                    print("Error:", file, e)

print("Features extracted:", len(features))


#model training now

# ------- Ye line se neeche ka code naya hai -------
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Data ko numpy array mein convert karo
X = np.array(features)
y = np.array(labels)

# Dataset split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network Model
model = MLPClassifier(hidden_layer_sizes=(150, 100), max_iter=2000, learning_rate='adaptive')

# Model train karo
model.fit(X_train, y_train)

# Predict karo test data pe
y_pred = model.predict(X_test)

# Accuracy check karo
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


import joblib
joblib.dump(model, "emotion_model.pkl")

