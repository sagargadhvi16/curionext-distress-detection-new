import numpy as np
import librosa

def extract_feature_chunk(audio_chunk, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=40).T
    chroma = librosa.feature.chroma_stft(y=audio_chunk, sr=sr).T
    mel = librosa.feature.melspectrogram(y=audio_chunk, sr=sr).T
    contrast = librosa.feature.spectral_contrast(y=audio_chunk, sr=sr).T
    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(audio_chunk), sr=sr
    ).T

    def stats(x):
        return np.hstack([
            np.mean(x, axis=0),
            np.std(x, axis=0),
            np.min(x, axis=0),
            np.max(x, axis=0),
        ])

    features = np.hstack([
        stats(mfcc),
        stats(chroma),
        stats(mel),
        stats(contrast),
        stats(tonnetz),
    ])
    return features.flatten()
