"""Unified audio feature aggregation."""

import numpy as np
from typing import Dict, Any

from src.audio.features import (
    extract_mfcc,
    extract_spectral_features,
    extract_prosodic_features,
)
from src.audio.encoder import YAMNetExtractor


class AudioFeatureExtractor:
    """
    Unified audio feature extractor.

    Combines:
    - MFCC features
    - Spectral features
    - Prosodic features
    - YAMNet embeddings
    """

    def __init__(self, use_yamnet: bool = True):
        """
        Args:
            use_yamnet: Whether to extract YAMNet embeddings
        """
        self.use_yamnet = use_yamnet
        self.yamnet = YAMNetExtractor() if use_yamnet else None

    def extract_all(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> Dict[str, Any]:
        """
        Extract all audio features.

        Args:
            audio: Mono audio waveform
            sr: Sample rate (default 16kHz)

        Returns:
            Dictionary containing all extracted features
        """
        if audio.size == 0:
            raise ValueError("Cannot extract features from empty audio")

        if audio.ndim != 1:
            raise ValueError("AudioFeatureExtractor expects mono audio")

        features: Dict[str, Any] = {}

        # MFCC (39-D)
        features["mfcc"] = extract_mfcc(audio, sr)

        # Spectral features (dict)
        features["spectral"] = extract_spectral_features(audio, sr)

        # Prosodic features (dict)
        features["prosodic"] = extract_prosodic_features(audio, sr)

        # YAMNet embeddings (1024-D)
        if self.use_yamnet:
            features["yamnet"] = self.yamnet.extract(audio, sr)

        return features
