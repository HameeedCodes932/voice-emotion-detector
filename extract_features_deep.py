import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfccs.shape[1] < 130:
        pad_width = 130 - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)))
    return mfccs

def load_data(folder_path):
    X, y = [], []
    for actor in os.listdir(folder_path):
        actor_path = os.path.join(folder_path, actor)
        for file in os.listdir(actor_path):
            part = file.split("-")[2]  # Emotion code
            label = emotion_map.get(part)
            if label:
                path = os.path.join(actor_path, file)
                mfcc = extract_features(path)
                X.append(mfcc)
                y.append(list(emotion_map.values()).index(label))
    return np.array(X), to_categorical(np.array(y))
