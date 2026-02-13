"""Temporal (windowed) audio feature extraction."""

import numpy as np
from typing import List, Dict, Callable


class WindowedFeatureExtractor:
    """
    Extract temporal feature sequences using sliding windows.

    Default:
    - Window size: 2 seconds
    - Overlap: 50%
    """

    def __init__(
        self,
        sr: int = 16000,
        window_sec: float = 2.0,
        overlap: float = 0.5,
    ):
        """
        Args:
            sr: Sample rate
            window_sec: Window length in seconds
            overlap: Fractional overlap between windows (0–1)
        """
        self.sr = sr
        self.window_size = int(window_sec * sr)
        self.hop_size = int(self.window_size * (1 - overlap))

        if self.hop_size <= 0:
            raise ValueError("Invalid overlap value")

    def extract(
        self,
        audio: np.ndarray,
        feature_fn: Callable[[np.ndarray, int], Dict[str, float] | np.ndarray],
    ) -> List:
        """
        Apply feature extraction over sliding windows.

        Args:
            audio: 1D mono audio signal
            feature_fn: Feature extraction function
                        (e.g., extract_mfcc, extract_prosodic_features)

        Returns:
            List of feature vectors (one per window)
        """
        if audio.size == 0:
            raise ValueError("Cannot extract temporal features from empty audio")

        features = []

        for start in range(0, len(audio) - self.window_size + 1, self.hop_size):
            window = audio[start : start + self.window_size]
            feat = feature_fn(window, self.sr)
            features.append(feat)

        return features
