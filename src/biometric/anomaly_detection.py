"""
Biometric Anomaly Detection Module

Detects stress / distress anomalies using:
- HRV features
- Activity context
- Movement behavior

This is a RULE-BASED baseline.
Later this can be replaced by ML / sequence models.
"""

from typing import Dict, Any


# -------------------------------------------------
# 🚨 Main Anomaly Detection Function
# -------------------------------------------------
def detect_biometric_anomaly(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect anomalies in biometric features.

    Args:
        features: Unified biometric feature dictionary
                  (output of biometric_aggregator)

    Returns:
        Dictionary with:
        - anomaly_detected (bool)
        - anomaly_score (0–1)
        - reasons (list of strings)
    """

    anomaly_score = 0.0
    reasons = []

    # -------------------------------------------------
    # ❤️ HRV BASED RULES
    # -------------------------------------------------
    rmssd = features.get("RMSSD", 0)
    lf_hf = features.get("LF_HF", 0)

    # Low HRV → stress indicator
    if rmssd > 0 and rmssd < 30:
        anomaly_score += 0.3
        reasons.append("Low HRV (RMSSD)")

    # Sympathetic dominance
    if lf_hf > 2.5:
        anomaly_score += 0.3
        reasons.append("High LF/HF ratio")

    # -------------------------------------------------
    # 🏃 ACTIVITY CONTEXT RULES
    # -------------------------------------------------
    activity = features.get("activity_level", "unknown")
    restless = features.get("restless", False)
    stillness = features.get("stillness", False)

    # Restlessness during sedentary activity → abnormal
    if activity == "sedentary" and restless:
        anomaly_score += 0.2
        reasons.append("Restless during sedentary activity")

    # Complete stillness + low HRV → possible fear / freeze
    if activity == "sedentary" and stillness and rmssd < 25:
        anomaly_score += 0.2
        reasons.append("Freeze response (low HRV + stillness)")

    # -------------------------------------------------
    # 🚑 FALL CONTEXT
    # -------------------------------------------------
    if features.get("fall_detected", False):
        anomaly_score = 1.0
        reasons.append("Fall detected")

    # -------------------------------------------------
    # 🧮 Final decision
    # -------------------------------------------------
    anomaly_detected = anomaly_score >= 0.5

    return {
        "anomaly_detected": anomaly_detected,
        "anomaly_score": round(min(anomaly_score, 1.0), 2),
        "reasons": reasons
    }


# -------------------------------------------------
# 🧪 Local test
# -------------------------------------------------
if __name__ == "__main__":
    sample_features = {
        "RMSSD": 22,
        "LF_HF": 3.1,
        "activity_level": "sedentary",
        "restless": True,
        "stillness": False,
        "fall_detected": False
    }

    result = detect_biometric_anomaly(sample_features)
    print("\nAnomaly Detection Result:\n")
    for k, v in result.items():
        print(f"{k}: {v}")
