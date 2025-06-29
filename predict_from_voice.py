import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import joblib

# 🔁 Record your voice
def record_audio(filename="test.wav", duration=10, fs=22050):
    print("🎙️ Recording... bolna shuru karo:")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print("✅ Recording saved: test.wav")

# 🎯 Feature extraction
def extract_features_from_file(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
    return np.mean(mfcc.T, axis=0)

# 📥 Load model
model = joblib.load("emotion_model.pkl")

# 🎙️ Record and Predict
record_audio()

features = extract_features_from_file("test.wav").reshape(1, -1)
prediction = model.predict(features)[0]

print(f"\n🤖 Predicted Emotion: {prediction.upper()}")
