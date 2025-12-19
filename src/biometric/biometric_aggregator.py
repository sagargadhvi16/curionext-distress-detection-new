"""
Biometric Feature Aggregator

Combines:
- HRV features
- Activity level
- Fall detection
- Movement pattern analysis
- Per-child baseline normalization

Output:
- Unified biometric feature dictionary (ML-ready)
"""

import numpy as np
from typing import Dict, Optional

# HRV
from src.biometric.hrv import extract_all_hrv_features

# Accelerometer modules
from src.biometric.activity_model import predict_activity_from_accel
from src.biometric.fall_detection import detect_fall_from_accel
from src.biometric.movement_pattern import analyze_movement_pattern

# Baseline module
from src.biometric.baseline import (
    initialize_child_baseline,
    update_child_baseline,
    normalize_with_baseline
)


# =====================================================
# 🔗 MAIN AGGREGATION FUNCTION
# =====================================================
def extract_biometric_features(
    rr_intervals: Optional[np.ndarray],
    accel_data: Optional[np.ndarray],
    child_id: Optional[str] = None,
    update_baseline: bool = False
) -> Dict[str, object]:
    """
    Extract unified biometric feature vector.

    Args:
        rr_intervals: RR intervals (ms or sec)
        accel_data: Accelerometer data (N x 3)
        child_id: Unique child identifier
        update_baseline: Whether to update baseline using this data

    Returns:
        Dictionary of biometric features
    """

    features: Dict[str, object] = {}

    # -------------------------------------------------
    # ❤️ HRV FEATURES
    # -------------------------------------------------
    if rr_intervals is not None and len(rr_intervals) > 0:
        hrv_features = extract_all_hrv_features(rr_intervals)
    else:
        hrv_features = {
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
        }

    # -------------------------------------------------
    # 🧒 BASELINE NORMALIZATION (HRV)
    # -------------------------------------------------
    if child_id is not None:
        initialize_child_baseline(child_id)

        if update_baseline:
            update_child_baseline(child_id, hrv_features)

        hrv_features = normalize_with_baseline(child_id, hrv_features)

    features.update(hrv_features)

    # -------------------------------------------------
    # 🏃 ACCELEROMETER FEATURES
    # -------------------------------------------------
    if accel_data is not None and len(accel_data) > 0:

        # Activity Level
        try:
            features["activity_level"] = predict_activity_from_accel(accel_data)
        except Exception:
            features["activity_level"] = "unknown"

        # Fall Detection
        try:
            fall = detect_fall_from_accel(accel_data)
            features["fall_detected"] = bool(fall.get("fall_detected", False))
            features["peak_g"] = float(fall.get("peak_g", 0.0))
        except Exception:
            features["fall_detected"] = False
            features["peak_g"] = 0.0

        # Movement Pattern
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

    output = extract_biometric_features(
        rr_intervals=rr,
        accel_data=accel,
        child_id="child_001",
        update_baseline=True
    )

    print("\nUnified Biometric Feature Vector:\n")
    for k, v in output.items():
        print(f"{k}: {v}")
