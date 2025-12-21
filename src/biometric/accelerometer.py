"""
Accelerometer signal processing.

Provides:
- Magnitude computation
- Movement peak detection (returns COUNT)
- Accelerometer feature extraction
"""

import numpy as np
from scipy import signal
from typing import Dict


# -------------------------------------------------
# 1️⃣ Magnitude computation
# -------------------------------------------------
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
    return np.linalg.norm(accel_data, axis=1)


# -------------------------------------------------
# 2️⃣ Detect movement peaks (FIXED)
# -------------------------------------------------
def detect_movement_peaks(
    magnitude: np.ndarray,
    threshold: float = 1.5
) -> int:
    """
    Detect movement peaks in accelerometer magnitude.

    Returns:
        Peak COUNT (int) – stable, ML-friendly output
    """
    if magnitude is None or len(magnitude) < 5:
        return 0

    adaptive_threshold = np.mean(magnitude) + threshold * np.std(magnitude)

    peaks, _ = signal.find_peaks(
        magnitude,
        height=adaptive_threshold,
        distance=5
    )

    return int(len(peaks))


# -------------------------------------------------
# 3️⃣ Extract accelerometer features
# -------------------------------------------------
def extract_accelerometer_features(
    accel_data: np.ndarray,
    sampling_rate: int = 50
) -> Dict[str, float]:
    """
    Extract features from 3-axis accelerometer data.

    Features:
    - Magnitude statistics
    - Energy
    - Peak count
    - Orientation variance
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

    if len(accel_data) < sampling_rate:  # < 1 second data
        return features

    # -------------------------------------------------
    # High-pass filter (remove gravity)
    # -------------------------------------------------
    try:
        b, a = signal.butter(
            N=2,
            Wn=0.5 / (sampling_rate / 2),
            btype="highpass"
        )
        accel_filtered = signal.filtfilt(b, a, accel_data, axis=0)
    except Exception:
        accel_filtered = accel_data

    # -------------------------------------------------
    # Magnitude features
    # -------------------------------------------------
    magnitude = compute_magnitude(accel_filtered)
    if len(magnitude) == 0:
        return features

    features["ACC_MEAN_MAG"] = float(np.mean(magnitude))
    features["ACC_STD_MAG"] = float(np.std(magnitude))
    features["ACC_MAX_MAG"] = float(np.max(magnitude))

    # Energy (activity intensity)
    features["ACC_ENERGY"] = float(np.sum(magnitude ** 2) / len(magnitude))

    # -------------------------------------------------
    # Peak-based movement
    # -------------------------------------------------
    peak_count = detect_movement_peaks(magnitude)
    features["ACC_PEAK_COUNT"] = float(peak_count)

    # -------------------------------------------------
    # Orientation / posture variance
    # -------------------------------------------------
    axis_variances = np.var(accel_filtered, axis=0)
    features["ACC_ORIENT_VAR"] = float(np.mean(axis_variances))

    return features
