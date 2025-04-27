import numpy as np
import librosa
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 1. Load the trained model
model = load_model('model.keras')

# 2. Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Feature extraction
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfcc, axis=1)

    features = [
        chroma_stft,
        rms,
        spectral_centroid,
        spectral_bandwidth,
        rolloff,
        zero_crossing_rate,
        *mfcc_means
    ]

    return np.array(features)

# 3. Function to prepare data for the LSTM model
def prepare_single_data(features, window_size=5):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features.reshape(-1, 1)).flatten()

    windowed_features = []
    for i in range(len(features) - window_size):
        window = features[i:i+window_size]
        windowed_features.append(window)

    return np.expand_dims(windowed_features, axis=0)

# 4. Predict function
def predict(file_path):
    features = extract_features(file_path)
    processed_features = prepare_single_data(features)

    prediction = model.predict(processed_features)
    prediction = (prediction.flatten() >= 0.5).astype(int)

    if prediction == 1:
        print(f"The uploaded voice is predicted as: **REAL** 🟢")
    else:
        print(f"The uploaded voice is predicted as: **FAKE** 🔴")

# Example usage
file_path = '/Users/pranaymishra/Desktop/deepfake_detector/linus-to-musk-DEMO.mp3'  # <-- Change this to your local file path
predict(file_path)
