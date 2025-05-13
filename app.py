import requests
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import io
import pickle

# Set up Streamlit page config
st.set_page_config(
    page_title="Speech Emotion Recognizer",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for background and styling
def add_bg_and_style():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://www.shutterstock.com/image-vector/voice-assistant-concept-vector-sound-wave-1501076531"); 
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }

        .stApp > header, .stApp > footer {visibility: hidden;}

        .block-container {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
        }

        h1, h2, h3 {
            color: #4B8BBE;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


add_bg_and_style()

# Download model
def download_model(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

model_url = "https://github.com/Kratika-159/Smart-Speech-Analysis/raw/master/SER_by_NOR.pkl"
local_model = "SER_by_NOR.pkl"
download_model(model_url, local_model)

# Envelope function
def envelope(y, rate, threshold):
    if y.ndim > 1:
        y = y.flatten()
    y_abs = np.abs(y)
    mask = y_abs > threshold
    return mask

# Feature extraction
def extract_feature(y, mfcc, chroma, mel, sample_rate):
    
    if chroma:
        stft = np.abs(librosa.stft(y))
    result = np.array([])
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

# App main function
def main():
    with st.container():
        st.markdown("<h1>🎤 Speech Emotion Recognizer</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center'>Upload a `.wav` file and let AI detect the emotion in the speech.</p>", unsafe_allow_html=True)


        st.markdown("## 📁 Upload Audio File")
        audio_file = st.file_uploader("Upload your `.wav` file", type=["wav"])

        if audio_file:
            st.audio(audio_file, format='audio/wav')

        st.markdown("---")

        if st.button("🔍 Analyze Emotion"):
            if audio_file is not None:
                with st.spinner("Analyzing the emotion from the audio..."):
                    audio_data = audio_file.read()
                    y, sample_rate = sf.read(io.BytesIO(audio_data))
                    envelope_mask = envelope(y, rate=sample_rate, threshold=0.0005)
                    y_filtered = y[envelope_mask]
                    ans = []
                    features = extract_feature(y_filtered, mfcc=True, chroma=True, mel=True, sample_rate=sample_rate)
                    ans.append(features)
                    ans = np.array(ans)

                    Pkl_Filename = local_model
                    with open(Pkl_Filename, 'rb') as file:
                        model = pickle.load(file)

                    prediction = model.predict(ans)
                    emotion = prediction[0]

                st.success("✅ Emotion Recognized")
                st.markdown(f"<h2 style='color:#FF4B4B;'>{emotion}</h2>", unsafe_allow_html=True)
            else:
                st.warning("Please upload a WAV file to analyze.")

if __name__ == "__main__":
    main()
