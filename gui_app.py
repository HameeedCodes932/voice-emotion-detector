import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("emotion_model.h5")

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
emoji_map = {
    'neutral': "😐", 'calm': "😌", 'happy': "😄", 'sad': "😢",
    'angry': "😠", 'fearful': "😨", 'disgust': "🤢", 'surprised': "😲"
}

# ========== FUNCTIONS ==========
def record_voice(filename="test.wav", duration=5, fs=22050):
    status_label.config(text="🔴 Recording...", fg="#ff5555")
    app.update()
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, recording, fs)
    status_label.config(text="✅ Recorded", fg="#50fa7b")

def extract_features(file_path, max_pad_len=220):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc.T  # shape becomes (220, 40)
    except Exception as e:
        print("Error extracting features:", e)
        return None


def display_waveform_and_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        fig, axs = plt.subplots(2, 1, figsize=(5, 4), dpi=100)
        axs[0].plot(audio, color='skyblue')
        axs[0].set_title("Waveform", fontsize=10)
        axs[0].set_ylabel("Amplitude")

        axs[1].imshow(mfccs, interpolation='nearest', cmap='viridis', origin='lower', aspect='auto')
        axs[1].set_title("MFCC (13 Coefficients)", fontsize=10)
        axs[1].set_ylabel("Index")
        axs[1].set_xlabel("Frame")

        for widget in waveform_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=waveform_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    except Exception as e:
        print("Visualization error:", e)

def show_bar_chart(pred_array):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    emotion_subset = emotions[:len(pred_array)]
    bars = ax.barh(emotion_subset, pred_array, color="#00bfff")
    ax.set_xlim([0, 1])
    ax.set_xlabel("Confidence")
    ax.set_title("Emotion Prediction", fontsize=12)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width*100:.1f}%", va='center')

    for widget in chart_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def predict_emotion():
    record_voice()
    features = extract_features("test.wav")
    if features is not None:
        features = np.expand_dims(features, axis=0)  # ✅ shape: (1, 220, 40)

        prediction = model.predict(features)
        pred_index = np.argmax(prediction)
        pred_emotion = emotions[pred_index]
        confidence = prediction[0][pred_index] * 100

        result_label.config(text=f"{emoji_map[pred_emotion]}  {pred_emotion.upper()}  ({confidence:.2f}%)", fg="white")
        display_waveform_and_mfcc("test.wav")
        show_bar_chart(prediction[0])
    else:
        result_label.config(text="❌ Error during prediction")

# ========== GUI ==========

app = tk.Tk()
app.title("🎧 Voice Emotion Detector - Modern Edition")
app.geometry("1080x700")
app.configure(bg="#1e1e2f")

title_label = tk.Label(app, text="🎤 Voice Emotion Detector", font=("Segoe UI", 22, "bold"), bg="#1e1e2f", fg="#00ffd2")
title_label.pack(pady=20)

status_label = tk.Label(app, text="Status: Idle", font=("Segoe UI", 10), bg="#1e1e2f", fg="gray")
status_label.pack()

predict_button = tk.Button(app, text="🎙 Record & Predict", font=("Segoe UI", 13, "bold"), bg="#0078d7", fg="white",
                           padx=20, pady=10, command=predict_emotion)
predict_button.pack(pady=20)

result_label = tk.Label(app, text="🎧 Prediction will appear here", font=("Segoe UI", 16), bg="#1e1e2f", fg="white")
result_label.pack(pady=10)

# Graph Frames
graphs_frame = tk.Frame(app, bg="#1e1e2f")
graphs_frame.pack(pady=10, fill='both', expand=True)

waveform_frame = tk.Frame(graphs_frame, bg="#1e1e2f")
waveform_frame.pack(side='left', padx=20, fill='both', expand=True)

chart_frame = tk.Frame(graphs_frame, bg="#1e1e2f")
chart_frame.pack(side='right', padx=20, fill='both', expand=True)

app.mainloop()
