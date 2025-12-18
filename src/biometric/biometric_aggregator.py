"""
Biometric Feature Aggregator

Combines:
- HRV features
- Activity level
- Fall detection
- Movement pattern analysis

Output:
- Unified biometric feature dictionary
"""

import numpy as np
from typing import Dict, Optional

# HRV
from src.biometric.hrv import extract_all_hrv_features

# Accelerometer-based modules
from src.biometric.activity_model import predict_activity_from_accel
from src.biometric.fall_detection import detect_fall_from_accel
from src.biometric.movement_pattern import analyze_movement_pattern


# ---------------------------------------------------------------------
# 🔗 Main Aggregation Function
# ---------------------------------------------------------------------
def extract_biometric_features(
    rr_intervals: Optional[np.ndarray],
    accel_data: Optional[np.ndarray]
) -> Dict[str, object]:
    """
    Extract unified biometric feature vector.

    Args:
        rr_intervals: RR intervals (ms or sec) or None
        accel_data: Accelerometer data (N x 3) or None

    Returns:
        Dictionary of biometric features
    """

    features: Dict[str, object] = {}

    # -------------------------------------------------
    # ❤️ HRV FEATURES
    # -------------------------------------------------
    if rr_intervals is not None and len(rr_intervals) > 0:
        try:
            hrv_features = extract_all_hrv_features(rr_intervals)
            features.update(hrv_features)
        except Exception:
            features.update({
                "RMSSD": 0.0,
                "SDNN": 0.0,
                "pNN50": 0.0,
                "LF": 0.0,
                "HF": 0.0,
                "LF_HF": 0.0,
                "FFT_VLF": 0.0,
                "FFT_LF": 0.0,
                "FFT_HF": 0.0,
                "FFT_LF_HF": 0.0,
                "SAMPEN": 0.0,
                "SD1": 0.0,
                "SD2": 0.0,
            })
    else:
        # HRV missing
        features.update({
            "RMSSD": 0.0,
            "SDNN": 0.0,
            "pNN50": 0.0,
            "LF": 0.0,
            "HF": 0.0,
            "LF_HF": 0.0,
            "FFT_VLF": 0.0,
            "FFT_LF": 0.0,
            "FFT_HF": 0.0,
            "FFT_LF_HF": 0.0,
            "SAMPEN": 0.0,
            "SD1": 0.0,
            "SD2": 0.0,
        })

    # -------------------------------------------------
    # 🏃 ACCELEROMETER FEATURES
    # -------------------------------------------------
    if accel_data is not None and len(accel_data) > 0:
        try:
            # Activity level
            activity_level = predict_activity_from_accel(accel_data)
            features["activity_level"] = activity_level
        except Exception:
            features["activity_level"] = "unknown"

        try:
            # Fall detection
            fall_result = detect_fall_from_accel(accel_data)
            features["fall_detected"] = fall_result.get("fall_detected", False)
            features["peak_g"] = fall_result.get("peak_g", 0.0)
        except Exception:
            features["fall_detected"] = False
            features["peak_g"] = 0.0

        try:
            # Movement pattern
            movement = analyze_movement_pattern(accel_data)
            features["movement_pattern"] = movement.get("pattern", "unknown")
            features["posture_change"] = movement.get("posture_change", False)
            features["restless"] = bool(movement.get("restless", False))
            features["stillness"] = bool(movement.get("stillness", False))
        except Exception:
            features["movement_pattern"] = "unknown"
            features["posture_change"] = False
            features["restless"] = False
            features["stillness"] = False

    else:
        # Accelerometer missing
        features.update({
            "activity_level": "unknown",
            "fall_detected": False,
            "peak_g": 0.0,
            "movement_pattern": "unknown",
            "posture_change": False,
            "restless": False,
            "stillness": False,
        })

    return features


# ---------------------------------------------------------------------
# 🧪 Quick test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Fake test inputs
    rr = np.random.normal(800, 40, 60)
    accel = np.random.normal([0, 0, 1], 0.1, size=(300, 3))

    output = extract_biometric_features(rr, accel)
    print("\nUnified Biometric Feature Vector:\n")
    for k, v in output.items():
        print(f"{k}: {v}")
