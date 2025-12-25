"""Multi-modal fusion module."""
from src.fusion.pairing import (
    pair_multimodal_samples,
    AudioSample,
    BiometricSample,
    PairedSample,
    load_samples_from_directories
)

__all__ = [
    'pair_multimodal_samples',
    'AudioSample',
    'BiometricSample',
    'PairedSample',
    'load_samples_from_directories'
]
