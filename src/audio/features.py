"""Audio feature extraction (temporal-preserving)."""

import librosa
import numpy as np
from typing import Dict


def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 13
) -> np.ndarray:
    """
    Extract MFCC features from audio.

    Returns:
        MFCC + delta + delta-delta features
        Shape: (39, T)
    """
    if audio.size == 0:
        raise ValueError("Cannot extract MFCC from empty audio")

    if audio.ndim != 1:
        raise ValueError("MFCC extraction expects mono audio")

    try:
        # MFCCs: (13, T)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # Temporal derivatives
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Stack along feature axis → (39, T)
        features = np.concatenate([mfcc, delta, delta2], axis=0)

        return features

    except Exception as e:
        raise ValueError(f"Failed to extract MFCC features: {e}")


def extract_spectral_features(
    audio: np.ndarray,
    sr: int
) -> Dict[str, np.ndarray]:
    """
    Extract spectral features while preserving time.

    Returns:
        Dict of temporal spectral features
    """
    if audio.size == 0:
        raise ValueError("Cannot extract spectral features from empty audio")

    try:
        n_mels = 64

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel, ref=np.max)      # (64, T)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)  # (12, T)

        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)  # (1, T)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)    # (1, T)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)  # (7, T)

        return {
            "mel_spectrogram": log_mel,
            "chroma": chroma,
            "spectral_centroid": centroid,
            "spectral_rolloff": rolloff,
            "spectral_contrast": contrast,
        }

    except Exception as e:
        raise ValueError(f"Failed to extract spectral features: {e}")


def extract_zero_crossing_rate(audio: np.ndarray) -> np.ndarray:
    """
    Extract zero-crossing rate (temporal).

    Returns:
        ZCR with shape (1, T)
    """
    if audio.size == 0:
        raise ValueError("Cannot extract ZCR from empty audio")

    try:
        return librosa.feature.zero_crossing_rate(audio)

    except Exception as e:
        raise ValueError(f"Failed to extract zero-crossing rate: {e}")


def extract_prosodic_features(
    audio: np.ndarray,
    sr: int
) -> Dict[str, np.ndarray]:
    """
    Extract prosodic features (temporal).

    Returns:
        Dict with time-varying pitch, energy, ZCR
    """
    if audio.size == 0:
        raise ValueError("Cannot extract prosodic features from empty audio")

    if audio.ndim != 1:
        raise ValueError("Prosodic feature extraction expects mono audio")

    try:
        # Pitch (F0) over time
        f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)

        # Energy over time
        rms = librosa.feature.rms(y=audio)  # (1, T)

        # Zero-crossing rate over time
        zcr = extract_zero_crossing_rate(audio)  # (1, T)

        return {
            "pitch_f0": f0,
            "energy": rms[0],
            "zcr": zcr[0],
        }

    except Exception as e:
        raise ValueError(f"Failed to extract prosodic features: {e}")
