import requests
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import io
import pickle

st.set_page_config(page_title="Speech Emotion Recognizer", page_icon="🎤", layout="centered")

# Function to download the model file
def download_model(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Download model if not already present
model_url = "https://github.com/Kratika-159/Smart-Speech-Analysis/raw/master/SER_by_NOR.pkl"
local_model = "SER_by_NOR.pkl"
download_model(model_url, local_model)

# Envelope Function
def envelope(y, rate, threshold):
    if y.ndim > 1:
        y = y.flatten()
    y_abs = np.abs(y)
    mask = y_abs > threshold
    return mask

# Feature Extraction Function
def extract_feature(y, mfcc, chroma, mel, sample_rate):
    result = np.array([])
    if chroma:
        stft = np.abs(librosa.stft(y))
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_feat))
    if mel:
        mel_feat = np.mean(librosa.feature.melspectrogram(y=y, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_feat))
    return result

# Streamlit App
def main():
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🎤 Speech Emotion Recognizer</h1>", unsafe_allow_html=True)
    st.markdown("Upload an audio clip and find out the emotion it conveys! Supports `.wav` files only.")

    with st.expander("ℹ️ How It Works"):
        st.write("""
        This tool uses machine learning to analyze the uploaded audio and identify the emotion expressed in the voice.
        It extracts key audio features like MFCCs, Chroma, and Mel Spectrogram, and feeds them into a trained model.
        """)

    st.markdown("## 📁 Upload Audio File")
    audio_file = st.file_uploader("Choose a `.wav` file", type=["wav"])

    if audio_file:
        st.audio(audio_file, format='audio/wav')
    
    st.markdown("---")

    if st.button("🔍 Analyze Emotion") and audio_file is not None:
        with st.spinner("Extracting features and predicting emotion..."):
            audio_data = audio_file.read()
            y, sample_rate = sf.read(io.BytesIO(audio_data))

            envelope_mask = envelope(y, rate=sample_rate, threshold=0.0005)
            y_filtered = y[envelope_mask]

            features = extract_feature(y_filtered, mfcc=True, chroma=True, mel=True, sample_rate=sample_rate)
            features = np.array([features])  # Reshape for prediction

            # Load model
            with open(local_model, 'rb') as file:  
                model = pickle.load(file)

            prediction = model.predict(features)
            emotion = prediction[0]

        st.success("✅ Emotion Identified!")
        st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>{emotion}</h2>", unsafe_allow_html=True)

    elif audio_file is None:
        st.info("Please upload a WAV file to begin.")

if __name__ == "__main__":
    main()
