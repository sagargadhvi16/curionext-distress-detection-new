"""Multi-modal fusion module."""
from src.fusion.pairing import (
    pair_multimodal_samples,
    AudioSample,
    BiometricSample,
    PairedSample,
    load_samples_from_directories
)
from src.fusion.rule_engine import RuleEngine, AlertLevel, Alert
from src.fusion.explainer import (
    visualize_attention_weights,
    calculate_confidence,
    DistressExplainer
)

__all__ = [
    'pair_multimodal_samples',
    'AudioSample',
    'BiometricSample',
    'PairedSample',
    'load_samples_from_directories',
    'RuleEngine',
    'AlertLevel',
    'Alert',
    'visualize_attention_weights',
    'calculate_confidence',
    'DistressExplainer'
]
