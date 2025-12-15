"""Generate synthetic biometric data for testing."""
import numpy as np
import json
from pathlib import Path


def generate_distress_hrv(duration: int = 60) -> dict:
    """
    Generate synthetic HRV data for distressed state.

    Distress characteristics:
    - Higher heart rate (lower RR intervals)
    - Higher variability

    Args:
        duration: Duration in seconds

    Returns:
        Dictionary with RR intervals
    """
    # Simulate faster, more variable heartbeat
    base_rr = 600  # ms (100 bpm)
    num_beats = int(duration * 1000 / base_rr)
    rr_intervals = base_rr + np.random.normal(0, 80, num_beats)
    rr_intervals = np.clip(rr_intervals, 400, 800).tolist()

    return {
        "rr_intervals": rr_intervals,
        "duration_sec": duration,
        "label": "distress"
    }


def generate_normal_hrv(duration: int = 60) -> dict:
    """
    Generate synthetic HRV data for normal state.

    Normal characteristics:
    - Normal heart rate
    - Lower variability
    """
    base_rr = 800  # ms (75 bpm)
    num_beats = int(duration * 1000 / base_rr)
    rr_intervals = base_rr + np.random.normal(0, 30, num_beats)
    rr_intervals = np.clip(rr_intervals, 700, 900).tolist()

    return {
        "rr_intervals": rr_intervals,
        "duration_sec": duration,
        "label": "normal"
    }


def generate_accelerometer_data(duration: int = 60, distress: bool = False) -> dict:
    """
    Generate synthetic accelerometer data.

    Args:
        duration: Duration in seconds
        distress: Whether to simulate distressed movement

    Returns:
        Dictionary with accelerometer data
    """
    sr = 50  # 50 Hz sampling
    num_samples = duration * sr

    if distress:
        # More erratic movement
        x = 0.5 * np.random.randn(num_samples)
        y = 0.5 * np.random.randn(num_samples)
        z = 9.8 + 0.8 * np.random.randn(num_samples)
    else:
        # Calmer movement
        x = 0.1 * np.random.randn(num_samples)
        y = 0.1 * np.random.randn(num_samples)
        z = 9.8 + 0.2 * np.random.randn(num_samples)

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "z": z.tolist(),
        "sampling_rate": sr,
        "duration_sec": duration,
        "label": "distress" if distress else "normal"
    }


if __name__ == "__main__":
    print("Generating synthetic biometric data...")

    # Create output directories
    hrv_dir = Path("data/synthetic/hrv")
    accel_dir = Path("data/synthetic/accelerometer")
    hrv_dir.mkdir(parents=True, exist_ok=True)
    accel_dir.mkdir(parents=True, exist_ok=True)

    # Generate distress samples
    for i in range(10):
        # HRV
        hrv_data = generate_distress_hrv()
        with open(hrv_dir / f"distress_{i:03d}.json", "w") as f:
            json.dump(hrv_data, f, indent=2)

        # Accelerometer
        accel_data = generate_accelerometer_data(distress=True)
        with open(accel_dir / f"distress_{i:03d}.json", "w") as f:
            json.dump(accel_data, f, indent=2)

    # Generate normal samples
    for i in range(10):
        # HRV
        hrv_data = generate_normal_hrv()
        with open(hrv_dir / f"normal_{i:03d}.json", "w") as f:
            json.dump(hrv_data, f, indent=2)

        # Accelerometer
        accel_data = generate_accelerometer_data(distress=False)
        with open(accel_dir / f"normal_{i:03d}.json", "w") as f:
            json.dump(accel_data, f, indent=2)

    print(f"Generated synthetic biometric data:")
    print(f"  - HRV: {hrv_dir}")
    print(f"  - Accelerometer: {accel_dir}")
