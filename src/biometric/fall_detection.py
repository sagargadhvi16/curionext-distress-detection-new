"""
Rule-based Fall Detection using Accelerometer data (KFall Dataset)

Logic:
1. Detect high impact (> 3g)
2. Check reduced movement after impact (relaxed inactivity)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict


# --------------------------------------------------
# 1️⃣ Compute acceleration magnitude
# --------------------------------------------------
def compute_magnitude(accel_data: np.ndarray) -> np.ndarray:
    """
    accel_data: shape (n_samples, 3)
    """
    return np.linalg.norm(accel_data, axis=1)


# --------------------------------------------------
# 2️⃣ Core fall detection logic (TUNED)
# --------------------------------------------------
def detect_fall_from_accel(
    accel_data: np.ndarray,
    impact_threshold_g: float = 3.0,
    inactivity_std_threshold: float = 0.6,
    post_impact_window: int = 50
) -> Dict[str, float]:
    """
    Detect fall using impact + relaxed inactivity logic.

    Returns:
        dict with detection details
    """

    result = {
        "fall_detected": False,
        "peak_g": 0.0,
        "impact_index": -1,
        "post_impact_std": 0.0
    }

    if accel_data.shape[0] < post_impact_window:
        return result

    # Compute magnitude
    magnitude = compute_magnitude(accel_data)

    peak_g = float(np.max(magnitude))
    impact_index = int(np.argmax(magnitude))

    result["peak_g"] = peak_g
    result["impact_index"] = impact_index

    # Condition 1: High impact
    if peak_g < impact_threshold_g:
        return result

    # Condition 2: Reduced movement after impact
    post_start = impact_index
    post_end = min(impact_index + post_impact_window, len(magnitude))

    post_window = magnitude[post_start:post_end]

    post_std = float(np.std(post_window))
    result["post_impact_std"] = post_std

    if post_std < inactivity_std_threshold:
        result["fall_detected"] = True

    return result


# --------------------------------------------------
# 3️⃣ Load ONE sensor CSV file
# --------------------------------------------------
def load_sensor_csv(file_path: str) -> np.ndarray:
    df = pd.read_csv(file_path)

    accel_cols = ["AccX", "AccY", "AccZ"]
    if not all(col in df.columns for col in accel_cols):
        raise ValueError(f"Missing accelerometer columns in {file_path}")

    return df[accel_cols].values


# --------------------------------------------------
# 4️⃣ Run fall detection for one subject
# --------------------------------------------------
def run_fall_detection_for_subject(subject_id: str) -> None:
    base_path = f"data/raw/fall/sensor_data/{subject_id}"

    print(f"\n🔍 Processing subject: {subject_id}")
    print("📂 Files found:", os.listdir(base_path))

    for file in os.listdir(base_path):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(base_path, file)

        accel_data = load_sensor_csv(file_path)
        result = detect_fall_from_accel(accel_data)

        print(
            f"{file} → "
            f"Fall: {result['fall_detected']} | "
            f"Peak G: {result['peak_g']:.2f} | "
            f"Post STD: {result['post_impact_std']:.2f}"
        )


# --------------------------------------------------
# 5️⃣ Entry point
# --------------------------------------------------
if __name__ == "__main__":
    run_fall_detection_for_subject("SA06")
