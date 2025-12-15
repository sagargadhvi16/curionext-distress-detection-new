"""Audio feature extraction."""
import librosa
import numpy as np
from typing import Dict


def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 40
) -> np.ndarray:
    """
    Extract MFCC features from audio.

    Args:
        audio: Audio signal array
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients

    Returns:
        MFCC feature array (n_mfcc, time_steps)

    TODO: Implement MFCC extraction
    """
    pass  # To be implemented by Intern 1


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

    TODO: Implement spectral feature extraction
    """
    pass  # To be implemented by Intern 1


def extract_zero_crossing_rate(audio: np.ndarray) -> np.ndarray:
    """
    Extract zero-crossing rate from audio.

    Args:
        audio: Audio signal array

    Returns:
        Zero-crossing rate array

    TODO: Implement ZCR extraction
    """
    pass  # To be implemented by Intern 1
