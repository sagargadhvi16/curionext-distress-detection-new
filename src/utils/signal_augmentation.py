"""Signal augmentation functions adapted from WER-SSL for audio and biometric signals."""

import cv2
import math
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from typing import Tuple, Union


# ============================================================================
# Noise Augmentation
# ============================================================================

def add_noise_with_snr(signal_data: np.ndarray, target_snr_db: float = 20.0) -> np.ndarray:
    """
    Add Gaussian noise to signal with target SNR.
    
    Adapted from WER-SSL.
    
    Args:
        signal_data: Input signal
        target_snr_db: Target Signal-to-Noise Ratio in dB
    
    Returns:
        Noisy signal
    """
    # Calculate signal power
    x_watts = signal_data ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts + 1e-10)
    
    # Calculate noise power for target SNR
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    
    # Generate Gaussian noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal_data))
    
    # Add noise
    noisy_signal = signal_data + noise
    return noisy_signal


# ============================================================================
# Temporal Warping Augmentation
# ============================================================================

def generate_random_curves(X: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
    """
    Generate random curves for magnitude warping.
    
    Adapted from WER-SSL.
    
    Args:
        X: Input signal
        sigma: Standard deviation for random curve generation
        knot: Number of knots for cubic spline
    
    Returns:
        Random curve coefficients
    """
    # Create knot points
    xx = np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1))
    
    # Generate random y-values for knots
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
    
    # Create x-range for interpolation
    x_range = np.arange(X.shape[0])
    
    # Cubic spline interpolation
    cs = CubicSpline(xx, yy)
    return cs(x_range)


def magnitude_warp(X: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    """
    Apply magnitude warping to signal.
    
    Warps the magnitude of the signal randomly over time.
    
    Args:
        X: Input signal
        sigma: Standard deviation for warping
    
    Returns:
        Magnitude-warped signal
    """
    return X * generate_random_curves(X, sigma)


def time_warp(
    signal_data: np.ndarray,
    sampling_freq: float = 100.0,
    pieces: int = 4,
    stretch_factor: float = 1.2,
    squeeze_factor: float = 0.8
) -> np.ndarray:
    """
    Apply time warping to signal by stretching and squeezing segments.
    
    Adapted from WER-SSL's time_warp_v3.
    
    Args:
        signal_data: Input signal
        sampling_freq: Sampling frequency (Hz)
        pieces: Number of segments along time axis
        stretch_factor: Factor to stretch selected segments
        squeeze_factor: Factor to squeeze other segments
    
    Returns:
        Time-warped signal (same length as input)
    """
    total_time = len(signal_data) / sampling_freq
    segment_time = total_time / pieces
    segment_len = int(np.floor(segment_time * sampling_freq))
    
    # Randomly select which segments to stretch
    sequence = list(range(0, pieces))
    stretch_indices = np.random.choice(
        sequence, 
        size=math.ceil(len(sequence) / 2), 
        replace=False
    )
    squeeze_indices = list(set(sequence) - set(stretch_indices))
    
    # Process each segment
    time_warped_segments = []
    for i in sequence:
        start_idx = int(i * segment_len)
        end_idx = min(int((i + 1) * segment_len), len(signal_data))
        segment = signal_data[start_idx:end_idx]
        
        segment = segment.reshape(-1, 1)
        
        if i in stretch_indices:
            new_length = int(np.ceil(len(segment) * stretch_factor))
        else:
            new_length = int(np.ceil(len(segment) * squeeze_factor))
        
        # Resize using OpenCV interpolation
        warped_segment = cv2.resize(segment, (1, new_length), interpolation=cv2.INTER_LINEAR)
        time_warped_segments.append(warped_segment.flatten())
    
    # Concatenate and ensure original length
    time_warped = np.concatenate(time_warped_segments)
    
    # Resample to original length if needed
    if len(time_warped) != len(signal_data):
        time_warped = signal.resample(time_warped, len(signal_data))
    
    return time_warped


# ============================================================================
# Permutation Augmentation
# ============================================================================

def permute_segments(
    signal_data: np.ndarray,
    num_pieces: int = 4
) -> np.ndarray:
    """
    Randomly permute temporal segments of signal.
    
    Adapted from WER-SSL.
    
    Args:
        signal_data: Input signal
        num_pieces: Number of segments to create and permute
    
    Returns:
        Signal with permuted segments
    """
    segments = np.array_split(signal_data, num_pieces)
    order = list(range(0, num_pieces))
    np.random.shuffle(order)
    
    permuted_signal = np.concatenate([segments[i] for i in order])
    return permuted_signal


# ============================================================================
# Crop and Resize Augmentation
# ============================================================================

def crop_and_resize(
    X: np.ndarray,
    num_pieces: int = 4,
    resample_to_original: bool = True
) -> np.ndarray:
    """
    Randomly crop and resize a segment of the signal.
    
    Adapted from WER-SSL's CropResize function.
    
    Args:
        X: Input signal
        num_pieces: Number of pieces to divide signal into
        resample_to_original: Whether to resample back to original length
    
    Returns:
        Signal with cropped and resized segment
    """
    piece_length = len(X) // num_pieces
    
    # Randomly select a piece
    selected_piece_idx = np.random.randint(0, num_pieces)
    start_idx = selected_piece_idx * piece_length
    end_idx = (selected_piece_idx + 1) * piece_length if selected_piece_idx < num_pieces - 1 else len(X)
    
    selected_piece = X[start_idx:end_idx]
    
    # Resample to original length
    if resample_to_original:
        resampled = signal.resample(selected_piece, len(X))
        return resampled
    else:
        return selected_piece


# ============================================================================
# Augmentation Pipeline
# ============================================================================

class SignalAugmentation:
    """
    Signal augmentation pipeline for audio and biometric data.
    
    Provides multiple augmentation strategies that can be composed.
    """
    
    def __init__(
        self,
        augmentation_prob: float = 0.5,
        noise_snr_range: Tuple[float, float] = (15.0, 30.0),
        magnitude_warp_sigma: float = 0.2,
        time_warp_pieces: int = 4,
        permute_pieces: int = 4,
        crop_pieces: int = 4,
        sampling_freq: float = 100.0
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentation_prob: Probability of applying each augmentation
            noise_snr_range: (min_snr, max_snr) for noise augmentation
            magnitude_warp_sigma: Sigma for magnitude warping
            time_warp_pieces: Number of pieces for time warping
            permute_pieces: Number of pieces for permutation
            crop_pieces: Number of pieces for crop-resize
            sampling_freq: Sampling frequency in Hz
        """
        self.augmentation_prob = augmentation_prob
        self.noise_snr_min, self.noise_snr_max = noise_snr_range
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.time_warp_pieces = time_warp_pieces
        self.permute_pieces = permute_pieces
        self.crop_pieces = crop_pieces
        self.sampling_freq = sampling_freq
    
    def apply_single(self, x: np.ndarray, augmentation_type: str = "random") -> np.ndarray:
        """
        Apply a single augmentation.
        
        Args:
            x: Input signal
            augmentation_type: Type of augmentation or 'random'
        
        Returns:
            Augmented signal
        """
        if augmentation_type == "noise":
            snr = np.random.uniform(self.noise_snr_min, self.noise_snr_max)
            return add_noise_with_snr(x, snr)
        
        elif augmentation_type == "magnitude_warp":
            return magnitude_warp(x, self.magnitude_warp_sigma)
        
        elif augmentation_type == "time_warp":
            return time_warp(x, self.sampling_freq, self.time_warp_pieces)
        
        elif augmentation_type == "permute":
            return permute_segments(x, self.permute_pieces)
        
        elif augmentation_type == "crop_resize":
            return crop_and_resize(x, self.crop_pieces)
        
        elif augmentation_type == "random":
            # Randomly select augmentation type
            aug_types = ["noise", "magnitude_warp", "time_warp", "permute", "crop_resize"]
            selected_type = np.random.choice(aug_types)
            return self.apply_single(x, selected_type)
        
        else:
            return x
    
    def apply_batch(
        self,
        x: np.ndarray,
        augmentation_type: str = "random",
        num_augmentations: int = 1
    ) -> list:
        """
        Apply augmentations to create multiple versions of signal.
        
        Args:
            x: Input signal
            augmentation_type: Type of augmentation ('random' for variety)
            num_augmentations: Number of augmented versions to create
        
        Returns:
            List of augmented signals (including original)
        """
        augmented_signals = [x]  # Include original
        
        for _ in range(num_augmentations):
            if np.random.random() < self.augmentation_prob:
                aug_signal = self.apply_single(x, augmentation_type)
                augmented_signals.append(aug_signal)
            else:
                augmented_signals.append(x.copy())
        
        return augmented_signals
    
    def apply_composition(
        self,
        x: np.ndarray,
        augmentation_types: list = None,
        apply_probability: float = 1.0
    ) -> np.ndarray:
        """
        Apply a composition of multiple augmentations.
        
        Args:
            x: Input signal
            augmentation_types: List of augmentation types to apply in sequence
            apply_probability: Probability of applying each augmentation
        
        Returns:
            Signal with composed augmentations
        """
        if augmentation_types is None:
            augmentation_types = ["noise", "magnitude_warp", "time_warp"]
        
        augmented = x.copy()
        
        for aug_type in augmentation_types:
            if np.random.random() < apply_probability:
                augmented = self.apply_single(augmented, aug_type)
        
        return augmented


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Normalize signal to [-1, 1] range."""
    mean = np.mean(x)
    std = np.std(x)
    if std > 0:
        return (x - mean) / (std + 1e-8)
    else:
        return x


def validate_signal(x: np.ndarray, expected_length: int = None) -> bool:
    """
    Validate signal format.
    
    Args:
        x: Signal to validate
        expected_length: Expected signal length (optional)
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(x, np.ndarray):
        return False
    
    if x.ndim != 1:
        return False
    
    if len(x) == 0:
        return False
    
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return False
    
    if expected_length is not None and len(x) != expected_length:
        return False
    
    return True
