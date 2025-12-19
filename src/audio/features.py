"""Audio feature extraction."""
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

    Args:
        audio: Audio signal array
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients

    Returns:
        39-dimensional feature vector
        (13 MFCC + 13 delta + 13 delta-delta)
    """
    if audio.size == 0:
        raise ValueError("Cannot extract MFCC from empty audio")

    if audio.ndim != 1:
        raise ValueError("MFCC extraction expects mono audio")

    try:
        # MFCCs: (13, T)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc
        )

        # First-order delta
        delta = librosa.feature.delta(mfcc)

        # Second-order delta-delta
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Mean pooling across time
        mfcc_mean = np.mean(mfcc, axis=1)
        delta_mean = np.mean(delta, axis=1)
        delta2_mean = np.mean(delta2, axis=1)

        # Concatenate → (39,)
        features = np.concatenate(
            [mfcc_mean, delta_mean, delta2_mean],
            axis=0
        )

        return features

    except Exception as e:
        raise ValueError(f"Failed to extract MFCC features: {e}")


def extract_spectral_features(
    audio: np.ndarray,
    sr: int
) -> Dict[str, np.ndarray]:
    """
    Extract spectral features (centroid, rolloff, contrast).

    Args:
        audio: Audio signal array
        sr: Sample rate

    Returns:
        Dictionary of spectral features

    """
    if audio.size == 0:
        raise ValueError("Cannot extract spectral features from empty audio")
    
    try:
        n_mels=64 # balanced default for 16kHz audio

        # Mel spectrogram → log-mel
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

        # Classic spectral features
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

        return {
            "mel_spectrogram": np.mean(log_mel, axis=1),
            "chroma": np.mean(chroma, axis=1),
            "spectral_centroid": np.mean(centroid),
            "spectral_rolloff": np.mean(rolloff),
            "spectral_contrast": np.mean(contrast, axis=1),
        }

    except Exception as e:
        raise ValueError(f"Failed to extract spectral features: {e}")


def extract_zero_crossing_rate(audio: np.ndarray) -> np.ndarray:
    """
    Extract zero-crossing rate from audio.

    Args:
        audio: Audio signal array

    Returns:
        Zero-crossing rate array

    """
    if audio.size == 0:
        raise ValueError("Cannot extract ZCR from empty audio")

    try:
        zcr = librosa.feature.zero_crossing_rate(audio)
        return np.mean(zcr)

    except Exception as e:
        raise ValueError(f"Failed to extract zero-crossing rate: {e}")
    


def extract_prosodic_features(audio:np.ndarray,sr:int)->Dict[str,float]:
    """
    Extract prosodic features from audio.

    Prosodic features describe pitch, energy, and temporal dynamics.
    """
    if audio.size==0:
        raise ValueError("Cannot extract prosodic features from empty audio")
    if audio.ndim != 1:
        raise ValueError("Prosodic feature extraction expects mono audio")

    try:
        # Pitch (F0) -measures fundamental freq
        f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
        pitch = float(np.mean(f0[f0 > 0])) if np.any(f0 > 0) else 0.0

        # Energy
        rms = librosa.feature.rms(y=audio)
        energy = float(np.mean(rms))

        # Zero Crossing Rate (reuse helper)
        zcr = extract_zero_crossing_rate(audio)

        # Spectral centroid & rolloff (reuse spectral extractor)
        spectral = extract_spectral_features(audio, sr)

        return{
            "pitch_f0":pitch,
            "energy":energy,
            "zcr":float(zcr),
            "spectral_centroid":float(spectral["spectral_centroid"]),
            "spectral_rolloff": float(spectral["spectral_rolloff"]),
        }
    except Exception as e:
        raise ValueError(f"Failed to extract prosodic features: {e}")
