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
from src.fusion.loss import MultiTaskLoss, create_multitask_loss
from src.fusion.training import train_epoch, validate_epoch
from src.fusion.scheduler import AdaptiveLRScheduler, create_scheduler
from src.fusion.duration_handler import (
    split_audio_by_bio_duration,
    prepare_multimodal_input_with_duration,
    stack_audio_segments_for_model,
    aggregate_predictions,
    calculate_duration_from_features
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
    'DistressExplainer',
    'MultiTaskLoss',
    'create_multitask_loss',
    'train_epoch',
    'validate_epoch',
    'AdaptiveLRScheduler',
    'create_scheduler',
    'split_audio_by_bio_duration',
    'prepare_multimodal_input_with_duration',
    'stack_audio_segments_for_model',
    'aggregate_predictions',
    'calculate_duration_from_features'
]
