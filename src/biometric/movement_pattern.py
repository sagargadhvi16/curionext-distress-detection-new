"""
Movement Pattern & Posture Analysis using Accelerometer Data

This module analyzes accelerometer windows to detect:
- Restlessness / agitation
- Posture changes (orientation shifts)
- Prolonged stillness (freeze)

Rule-based, lightweight, and wearable-friendly.
"""

import numpy as np
from typing import Dict, Optional

# ---------------------------------------------------------------------
# 1️⃣ Axis-wise statistics
# ---------------------------------------------------------------------
def compute_axis_stats(accel_data: np.ndarray) -> Dict[str, float]:
    """
    Compute mean and variance for each accelerometer axis.

    accel_data: shape (n_samples, 3)
    """

    x = accel_data[:, 0]
    y = accel_data[:, 1]
    z = accel_data[:, 2]

    return {
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
        "mean_z": float(np.mean(z)),
        "var_x": float(np.var(x)),
        "var_y": float(np.var(y)),
        "var_z": float(np.var(z)),
    }


# ---------------------------------------------------------------------
# 2️⃣ Magnitude computation
# ---------------------------------------------------------------------
def compute_magnitude(accel_data: np.ndarray) -> np.ndarray:
    """
    Compute acceleration magnitude.
    """
    return np.linalg.norm(accel_data, axis=1)


# ---------------------------------------------------------------------
# 3️⃣ Detect posture change
# ---------------------------------------------------------------------
def detect_posture_change(
    prev_stats: Dict[str, float],
    curr_stats: Dict[str, float]
) -> bool:
    """
    Detect posture change via dominant axis shift.
    """

    def dominant_axis(stats):
        means = {
            "x": abs(stats["mean_x"]),
            "y": abs(stats["mean_y"]),
            "z": abs(stats["mean_z"]),
        }
        return max(means, key=means.get)

    prev_axis = dominant_axis(prev_stats)
    curr_axis = dominant_axis(curr_stats)

    return prev_axis != curr_axis


# ---------------------------------------------------------------------
# 4️⃣ Detect restlessness
# ---------------------------------------------------------------------
def detect_restlessness(magnitude: np.ndarray) -> bool:
    """
    Detect restless / agitated movement.
    """

    var_mag = np.var(magnitude)
    peak_count = np.sum(magnitude > (np.mean(magnitude) + 1.5 * np.std(magnitude)))

    return (var_mag > 0.25) or (peak_count > 8)


# ---------------------------------------------------------------------
# 5️⃣ Detect prolonged stillness
# ---------------------------------------------------------------------
def detect_stillness(magnitude: np.ndarray) -> bool:
    """
    Detect near-zero movement (freeze / inactivity).
    """

    return np.var(magnitude) < 0.01


# ---------------------------------------------------------------------
# 6️⃣ MAIN FUNCTION
# ---------------------------------------------------------------------
def analyze_movement_pattern(
    accel_data: np.ndarray,
    prev_window_stats: Optional[Dict[str, float]] = None
) -> Dict[str, object]:
    """
    Analyze movement pattern for a single accelerometer window.

    Returns:
        pattern analysis dictionary
    """

    if accel_data.ndim != 2 or accel_data.shape[1] != 3:
        raise ValueError("accel_data must have shape (n_samples, 3)")

    magnitude = compute_magnitude(accel_data)
    axis_stats = compute_axis_stats(accel_data)

    posture_change = False
    if prev_window_stats is not None:
        posture_change = detect_posture_change(prev_window_stats, axis_stats)

    restless = detect_restlessness(magnitude)
    stillness = detect_stillness(magnitude)

    # Decide main pattern
    if posture_change:
        pattern = "posture_change"
    elif restless:
        pattern = "restless"
    elif stillness:
        pattern = "stillness"
    else:
        pattern = "stable"

    return {
        "pattern": pattern,
        "posture_change": posture_change,
        "restless": restless,
        "stillness": stillness,
        "axis_stats": axis_stats,
        "magnitude_var": float(np.var(magnitude)),
    }
