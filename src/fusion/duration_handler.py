"""Duration mismatch handling utilities for multi-modal data."""
import torch
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def split_audio_by_bio_duration(
    audio_data: np.ndarray,
    audio_sr: int,
    bio_duration: float,
    audio_duration: Optional[float] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Split audio into aligned and tail segments based on biometric duration.
    
    Args:
        audio_data: Raw audio array (samples,)
        audio_sr: Audio sample rate
        bio_duration: Biometric data duration in seconds
        audio_duration: Optional audio duration (calculated if None)
        
    Returns:
        Tuple of (aligned_audio, tail_audio)
        - aligned_audio: Audio segment matching bio_duration
        - tail_audio: Remaining audio or None if no tail
        
    Example:
        >>> audio = np.random.randn(16000 * 60)  # 60 seconds at 16kHz
        >>> aligned, tail = split_audio_by_bio_duration(audio, 16000, 30.0)
        >>> print(f"Aligned: {len(aligned)/16000}s, Tail: {len(tail)/16000 if tail is not None else 0}s")
        Aligned: 30.0s, Tail: 30.0s
    """
    if audio_duration is None:
        audio_duration = len(audio_data) / audio_sr
    
    logger.info(f"Audio duration: {audio_duration:.2f}s, Bio duration: {bio_duration:.2f}s")
    
    # Calculate sample indices
    bio_samples = int(bio_duration * audio_sr)
    
    if audio_duration <= bio_duration:
        # Audio is shorter or equal, no tail
        logger.info("Audio duration <= bio duration, no tail segment")
        return audio_data, None
    
    # Split audio
    aligned_audio = audio_data[:bio_samples]
    tail_audio = audio_data[bio_samples:]
    
    logger.info(
        f"Split audio into aligned ({len(aligned_audio)/audio_sr:.2f}s) "
        f"and tail ({len(tail_audio)/audio_sr:.2f}s) segments"
    )
    
    return aligned_audio, tail_audio


def prepare_multimodal_input_with_duration(
    audio_data: np.ndarray,
    biometric_data: np.ndarray,
    audio_sr: int = 16000,
    audio_duration: Optional[float] = None,
    bio_duration: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Prepare multi-modal input handling duration mismatches.
    
    If audio duration > bio duration, splits audio into:
    - Aligned segment (matches bio duration)
    - Tail segment (remaining audio)
    
    Args:
        audio_data: Raw audio array
        biometric_data: Biometric feature array
        audio_sr: Audio sample rate
        audio_duration: Audio duration in seconds (calculated if None)
        bio_duration: Biometric duration in seconds (must be provided or inferred)
        
    Returns:
        Dictionary with keys:
        - 'audio_aligned': Audio segment aligned with biometric
        - 'audio_tail': Remaining audio (or None)
        - 'biometric': Biometric data
        - 'has_tail': Boolean indicating if tail exists
        - 'audio_duration': Total audio duration
        - 'bio_duration': Biometric duration
    """
    if audio_duration is None:
        audio_duration = len(audio_data) / audio_sr
    
    if bio_duration is None:
        # Try to infer from biometric data structure
        # This is a placeholder - in practice, duration should be provided
        bio_duration = audio_duration
        logger.warning(
            f"Bio duration not provided, assuming same as audio: {bio_duration:.2f}s"
        )
    
    # Split audio based on biometric duration
    aligned_audio, tail_audio = split_audio_by_bio_duration(
        audio_data, audio_sr, bio_duration, audio_duration
    )
    
    return {
        'audio_aligned': aligned_audio,
        'audio_tail': tail_audio,
        'biometric': biometric_data,
        'has_tail': tail_audio is not None,
        'audio_duration': audio_duration,
        'bio_duration': bio_duration
    }


def stack_audio_segments_for_model(
    audio_aligned: np.ndarray,
    audio_tail: Optional[np.ndarray],
    pad_tail: bool = True
) -> np.ndarray:
    """
    Stack aligned and tail audio segments for batch processing.
    
    Args:
        audio_aligned: Aligned audio segment
        audio_tail: Tail audio segment (or None)
        pad_tail: Whether to pad tail to match aligned length
        
    Returns:
        Stacked audio array:
        - If no tail: (1, len(audio_aligned))
        - If tail exists: (2, max_len) where first is aligned, second is tail
    """
    if audio_tail is None:
        # No tail, return aligned only
        return audio_aligned.reshape(1, -1)
    
    # Pad tail to match aligned length if needed
    if pad_tail and len(audio_tail) < len(audio_aligned):
        pad_length = len(audio_aligned) - len(audio_tail)
        audio_tail = np.pad(audio_tail, (0, pad_length), mode='constant')
    elif len(audio_tail) > len(audio_aligned):
        # Truncate tail if longer (shouldn't happen in normal cases)
        audio_tail = audio_tail[:len(audio_aligned)]
    
    # Stack segments
    return np.stack([audio_aligned, audio_tail], axis=0)


def aggregate_predictions(
    pred_aligned: Dict[str, torch.Tensor],
    pred_tail: Optional[Dict[str, torch.Tensor]],
    aggregation_method: str = 'max'
) -> Dict[str, torch.Tensor]:
    """
    Aggregate predictions from aligned and tail segments.
    
    Args:
        pred_aligned: Predictions from aligned segment
        pred_tail: Predictions from tail segment (or None)
        aggregation_method: How to aggregate ('max', 'mean', 'weighted')
        
    Returns:
        Aggregated predictions dictionary
    """
    if pred_tail is None:
        return pred_aligned
    
    aggregated = {}
    
    if aggregation_method == 'max':
        # Use maximum for distress detection (conservative approach)
        for key in ['distress_logits', 'severity', 'type_logits']:
            if key in pred_aligned and key in pred_tail:
                aggregated[key] = torch.max(
                    torch.stack([pred_aligned[key], pred_tail[key]]),
                    dim=0
                )[0]
    elif aggregation_method == 'mean':
        # Average predictions
        for key in ['distress_logits', 'severity', 'type_logits']:
            if key in pred_aligned and key in pred_tail:
                aggregated[key] = torch.mean(
                    torch.stack([pred_aligned[key], pred_tail[key]]),
                    dim=0
                )
    elif aggregation_method == 'weighted':
        # Weight by segment duration (aligned gets more weight)
        # Default weights: 0.6 for aligned, 0.4 for tail
        w_aligned, w_tail = 0.6, 0.4
        for key in ['distress_logits', 'severity', 'type_logits']:
            if key in pred_aligned and key in pred_tail:
                aggregated[key] = (
                    w_aligned * pred_aligned[key] + w_tail * pred_tail[key]
                )
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    # Store individual segment predictions for analysis
    aggregated['aligned_outputs'] = pred_aligned
    aggregated['tail_outputs'] = pred_tail
    
    return aggregated


def calculate_duration_from_features(
    features: np.ndarray,
    feature_type: str,
    sample_rate: Optional[int] = None
) -> Optional[float]:
    """
    Calculate duration from feature array based on feature type.
    
    Args:
        features: Feature array
        feature_type: Type of features ('audio', 'hrv', 'accelerometer')
        sample_rate: Sample rate for time-series features
        
    Returns:
        Duration in seconds or None if cannot determine
    """
    if feature_type == 'audio' and sample_rate:
        return len(features) / sample_rate
    elif feature_type == 'hrv':
        # RR intervals are in milliseconds
        if len(features) > 0:
            return np.sum(features) / 1000.0
        return None
    elif feature_type == 'accelerometer' and sample_rate:
        # Assume 3D accelerometer (x, y, z)
        if features.ndim == 2 and features.shape[1] == 3:
            return features.shape[0] / sample_rate
        return len(features) / sample_rate if sample_rate else None
    
    return None
