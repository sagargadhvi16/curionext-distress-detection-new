"""
Biometric Feature Aggregator

Combines:
- HRV features
- Activity level
- Fall detection
- Movement pattern analysis

Output:
- Unified biometric feature dictionary (stateless, ML-ready)
"""

import numpy as np
from typing import Dict, Optional

# HRV
from src.biometric.hrv import extract_all_hrv_features

# Accelerometer modules
from src.biometric.activity_model import predict_activity_from_accel
from src.biometric.fall_detection import detect_fall_from_accel
from src.biometric.movement_pattern import analyze_movement_pattern


# =====================================================
# 🔗 MAIN AGGREGATION FUNCTION
# =====================================================
def extract_biometric_features(
    rr_intervals: Optional[np.ndarray],
    accel_data: Optional[np.ndarray]
) -> Dict[str, object]:
    """
    Extract unified biometric feature vector.

    Args:
        rr_intervals: RR intervals (ms)
        accel_data: Accelerometer data (N x 3)

    Returns:
        Dictionary of biometric features
    """

    features: Dict[str, object] = {}

    # -------------------------------------------------
    # ❤️ HRV FEATURES
    # -------------------------------------------------
    if rr_intervals is not None and len(rr_intervals) > 0:
        try:
            features.update(extract_all_hrv_features(rr_intervals))
        except Exception:
            pass

    # Ensure HRV keys always exist
    features.setdefault("RMSSD", 0.0)
    features.setdefault("SDNN", 0.0)
    features.setdefault("pNN50", 0.0)
    features.setdefault("LF", 0.0)
    features.setdefault("HF", 0.0)
    features.setdefault("LF_HF", 0.0)
    features.setdefault("FFT_VLF", 0.0)
    features.setdefault("FFT_LF", 0.0)
    features.setdefault("FFT_HF", 0.0)
    features.setdefault("FFT_LF_HF", 0.0)
    features.setdefault("SAMPEN", 0.0)
    features.setdefault("SD1", 0.0)
    features.setdefault("SD2", 0.0)

    # -------------------------------------------------
    # 🏃 ACCELEROMETER FEATURES
    # -------------------------------------------------
    if accel_data is not None and len(accel_data) > 0:

        # Activity
        try:
            features["activity_level"] = predict_activity_from_accel(accel_data)
        except Exception:
            features["activity_level"] = "unknown"

        # Fall detection
        try:
            fall = detect_fall_from_accel(accel_data)
            features["fall_detected"] = bool(fall.get("fall_detected", False))
            features["peak_g"] = float(fall.get("peak_g", 0.0))
        except Exception:
            features["fall_detected"] = False
            features["peak_g"] = 0.0

        # Movement pattern
        try:
            movement = analyze_movement_pattern(accel_data)
            features["movement_pattern"] = movement.get("pattern", "unknown")
            features["posture_change"] = bool(movement.get("posture_change", False))
            features["restless"] = bool(movement.get("restless", False))
            features["stillness"] = bool(movement.get("stillness", False))
        except Exception:
            features.update({
                "movement_pattern": "unknown",
                "posture_change": False,
                "restless": False,
                "stillness": False,
            })

    else:
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


# =====================================================
# 🧪 QUICK LOCAL TEST
# =====================================================
if __name__ == "__main__":
    rr = np.random.normal(800, 40, 60)
    accel = np.random.normal([0, 0, 1], 0.1, size=(300, 3))

    output = extract_biometric_features(rr, accel)

    print("\nUnified Biometric Feature Vector:\n")
    for k, v in output.items():
        print(f"{k}: {v}")
