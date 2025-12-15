"""Audio preprocessing utilities."""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple


def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 16000 Hz for YAMNet)

    Returns:
        Tuple of (audio_data, sample_rate)

    TODO: Implement audio loading with error handling
    """
    pass  # To be implemented by Intern 1


def normalize_audio(audio: np.ndarray, method: str = "peak") -> np.ndarray:
    """
    Normalize audio using peak or RMS normalization.

    Args:
        audio: Audio signal array
        method: Normalization method ("peak" or "rms")

    Returns:
        Normalized audio array

    TODO: Implement normalization methods
    """
    pass  # To be implemented by Intern 1


def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: int = 20
) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.

    Args:
        audio: Audio signal array
        sr: Sample rate
        top_db: Threshold in dB below reference to consider as silence

    Returns:
        Trimmed audio array

    TODO: Implement silence trimming
    """
    pass  # To be implemented by Intern 1
