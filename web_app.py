# web_app.py - Full Working Streamlit Voice Emotion Detector

import os
import librosa
import numpy as np
import soundfile as sf
import sounddevice as sd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tempfile
from matplotlib.backends.backend_agg import RendererAgg
import threading

_lock = threading.Lock()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------- Load model ----------
@st.cache_resource
def load_emotion_model():
    return load_model("emotion_model.h5")

model = load_emotion_model()

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
emoji_map = {
    'neutral': "😐", 'calm': "😌", 'happy': "😄", 'sad': "😢",
    'angry': "😠", 'fearful': "😨", 'disgust': "🤢", 'surprised': "😲"
}

# ---------- Feature Extraction ----------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfccs.shape[1] < 220:
        pad_width = 220 - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)))
    elif mfccs.shape[1] > 220:
        mfccs = mfccs[:, :220]
    return mfccs.T, audio, sr

# ---------- Predict ----------
def predict_emotion(file_path):
    mfccs, audio, sr = extract_features(file_path)
    features = np.expand_dims(mfccs, axis=0)
    prediction = model.predict(features)[0]
    pred_index = np.argmax(prediction)
    emotion = emotions[pred_index]
    confidence = prediction[pred_index]
    return emotion, confidence, prediction, audio, sr

# ---------- Plot Waveform and MFCC ----------
def plot_waveform_and_mfcc(audio, sr):
    fig, axs = plt.subplots(2, 1, figsize=(8, 4), dpi=100)

    axs[0].plot(audio, color='skyblue')
    axs[0].set_title("Waveform")
    axs[0].set_ylabel("Amplitude")

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    axs[1].imshow(mfccs, cmap='viridis', origin='lower', aspect='auto')
    axs[1].set_title("MFCC (13 Coefficients)")
    axs[1].set_xlabel("Frame")
    axs[1].set_ylabel("MFCC Index")

    return fig

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Voice Emotion Detection", layout="centered")
st.title("🎧 Voice Emotion Detection Web App")
st.markdown("Upload or record audio to detect the speaker's emotion.")

# ---------- Record Audio ----------
duration = st.slider("🎛️ Recording duration (seconds)", 2, 10, 5)
if st.button("🎤 Record Audio"):
    fs = 22050
    with st.spinner("Recording..."):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, recording, fs)
            st.audio(tmpfile.name, format='audio/wav')
            emotion, conf, probs, audio, sr = predict_emotion(tmpfile.name)
            st.success(f"### Predicted: {emoji_map[emotion]} {emotion.upper()} ({conf * 100:.2f}%)")
            st.bar_chart({emotions[i]: probs[i] for i in range(len(emotions))})
            with _lock:
                fig = plot_waveform_and_mfcc(audio, sr)
                st.pyplot(fig)

# ---------- Upload Audio ----------
uploaded_file = st.file_uploader("📤 Or upload an audio file", type=["wav", "mp3", "ogg"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded_file.read())
        st.audio(tmpfile.name, format='audio/wav')
        emotion, conf, probs, audio, sr = predict_emotion(tmpfile.name)
        st.success(f"### Predicted: {emoji_map[emotion]} {emotion.upper()} ({conf * 100:.2f}%)")
        st.bar_chart({emotions[i]: probs[i] for i in range(len(emotions))})
        with _lock:
            fig = plot_waveform_and_mfcc(audio, sr)
            st.pyplot(fig)
