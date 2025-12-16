


"""Accelerometer signal processing."""

import numpy as np
from scipy import signal
from typing import Dict, Tuple


def compute_magnitude(accel_data: np.ndarray) -> np.ndarray:
    """
    Compute magnitude of 3-axis accelerometer signal.

    Args:
        accel_data: Array of shape (n_samples, 3)

    Returns:
        Magnitude array of shape (n_samples,)
    """
    if accel_data is None or accel_data.ndim != 2 or accel_data.shape[1] != 3:
        return np.array([])

    # √(x² + y² + z²)
    magnitude = np.linalg.norm(accel_data, axis=1)
    return magnitude


def detect_movement_peaks(
    magnitude: np.ndarray,
    threshold: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect movement peaks in accelerometer magnitude.

    Args:
        magnitude: Magnitude signal
        threshold: Peak detection threshold (relative to std)

    Returns:
        Tuple of (peak_indices, peak_values)
    """
    if magnitude is None or len(magnitude) < 5:
        return np.array([]), np.array([])

    # Adaptive threshold using signal statistics
    adaptive_threshold = np.mean(magnitude) + threshold * np.std(magnitude)

    peaks, properties = signal.find_peaks(
        magnitude,
        height=adaptive_threshold,
        distance=5
    )

    peak_values = properties.get("peak_heights", np.array([]))
    return peaks, peak_values


def extract_accelerometer_features(
    accel_data: np.ndarray,
    sampling_rate: int = 50
) -> Dict[str, float]:
    """
    Extract features from 3-axis accelerometer data.

    Features include:
    - Movement intensity (magnitude statistics)
    - Activity level (signal energy)
    - Postural changes (axis variance)
    """
    features = {
        "ACC_MEAN_MAG": 0.0,
        "ACC_STD_MAG": 0.0,
        "ACC_MAX_MAG": 0.0,
        "ACC_ENERGY": 0.0,
        "ACC_PEAK_COUNT": 0.0,
        "ACC_ORIENT_VAR": 0.0
    }

    if accel_data is None or accel_data.ndim != 2 or accel_data.shape[1] != 3:
        return features

    if len(accel_data) < sampling_rate:  # < 1 second of data
        return features

    # --- Step 1: High-pass filter to remove gravity ---
    try:
        b, a = signal.butter(
            N=2,
            Wn=0.5 / (sampling_rate / 2),
            btype="highpass"
        )
        accel_filtered = signal.filtfilt(b, a, accel_data, axis=0)
    except Exception:
        accel_filtered = accel_data

    # --- Step 2: Magnitude computation ---
    magnitude = compute_magnitude(accel_filtered)
    if len(magnitude) == 0:
        return features

    # --- Step 3: Magnitude-based features ---
    features["ACC_MEAN_MAG"] = float(np.mean(magnitude))
    features["ACC_STD_MAG"] = float(np.std(magnitude))
    features["ACC_MAX_MAG"] = float(np.max(magnitude))

    # Signal energy (activity level)
    features["ACC_ENERGY"] = float(np.sum(magnitude ** 2) / len(magnitude))

    # --- Step 4: Peak-based movement ---
    peaks, _ = detect_movement_peaks(magnitude)
    features["ACC_PEAK_COUNT"] = float(len(peaks))

    # --- Step 5: Orientation / posture variance ---
    # Variance across x,y,z indicates posture instability
    axis_variances = np.var(accel_filtered, axis=0)
    features["ACC_ORIENT_VAR"] = float(np.mean(axis_variances))

    return features
