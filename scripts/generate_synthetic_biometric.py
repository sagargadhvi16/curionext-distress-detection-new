"""
Synthetic Biometric Data Generator (Children 3–12 years)

Generates:
- HRV (RR intervals)
- Accelerometer (x, y, z)

States:
- normal
- distress

Design goals:
- Age-aware pediatric physiology (3–12 years)
- Realistic distress dynamics
- Fusion-ready structure
"""

import numpy as np
import json
from pathlib import Path


# ---------------------------------------------------------------------
# HRV HELPERS (3–12 YEARS)
# ---------------------------------------------------------------------

def _age_to_base_rr_ms(age_years: float, state: str = "normal") -> int:
    """
    Estimate base RR interval (ms) for children aged 3–12 years.
    Heuristic, not clinical.
    """
    age_years = float(np.clip(age_years, 3.0, 12.0))

    if age_years < 5.0:
        base_normal = 600   # ~100 bpm
    elif age_years < 8.0:
        base_normal = 680   # ~88 bpm
    elif age_years < 11.0:
        base_normal = 760   # ~79 bpm
    else:
        base_normal = 820   # ~73 bpm

    if state == "distress":
        # Younger children show stronger HR acceleration
        drop = 80 + int(20 * (12.0 - age_years) / 9.0)
        return max(350, base_normal - drop)

    return base_normal


def generate_distress_hrv(
    duration: int = 60,
    age_years: float = 7.0,
    activity_level: float = 0.5
) -> dict:
    """
    Generate HRV for distressed child (3–12 yrs).
    """
    base_rr = _age_to_base_rr_ms(age_years, state="distress")
    sd_ms = 60 + 40 * activity_level

    n_beats = max(1, int(duration * 1000 / base_rr))
    rr = base_rr + np.random.normal(0, sd_ms, n_beats)

    # Occasional ectopic-like spikes
    for _ in range(int(0.02 * n_beats)):
        rr[np.random.randint(0, n_beats)] += np.random.uniform(-150, 150)

    rr = np.clip(rr, 350, 1200).tolist()

    return {
        "rr_intervals": rr,
        "duration_sec": duration,
        "label": "distress",
    }


def generate_normal_hrv(
    duration: int = 60,
    age_years: float = 7.0,
    activity_level: float = 0.3
) -> dict:
    """
    Generate HRV for normal child (3–12 yrs).
    """
    base_rr = _age_to_base_rr_ms(age_years, state="normal")
    sd_ms = 20 + 10 * activity_level

    n_beats = max(1, int(duration * 1000 / base_rr))
    rr = base_rr + np.random.normal(0, sd_ms, n_beats)

    rr = np.clip(rr, 400, 1200).tolist()

    return {
        "rr_intervals": rr,
        "duration_sec": duration,
        "label": "normal",
    }


# ---------------------------------------------------------------------
# ACCELEROMETER HELPERS
# ---------------------------------------------------------------------

def _child_movement_envelope(
    num_samples: int,
    activity_level: float,
    distress: bool
) -> np.ndarray:
    """
    Bursty movement envelope typical of children.
    """
    env = np.zeros(num_samples)
    pos = 0

    while pos < num_samples:
        burst_len = int(np.random.uniform(0.1, 0.6) * 50 * (1 + activity_level))
        pause = int(np.random.uniform(0.05, 0.4) * 50)

        if burst_len <= 0:
            break

        end = min(num_samples, pos + burst_len)
        env[pos:end] = np.linspace(
            0.2,
            1.0 + 0.5 * int(distress),
            end - pos
        )
        pos += burst_len + pause

    return env


def generate_accelerometer_data(
    duration: int = 60,
    distress: bool = False,
    age_years: float = 7.0,
    activity_level: float = 0.5
) -> dict:
    """
    Generate accelerometer data for children aged 3–12 years.
    """
    sr = 50  # Hz
    num_samples = max(1, int(duration * sr))

    age_years = float(np.clip(age_years, 3.0, 12.0))
    # Younger children → more jitter
    age_factor = 1.1 - 0.03 * (age_years - 3.0)

    base_amp = (0.6 if distress else 0.15) * age_factor
    env = _child_movement_envelope(num_samples, activity_level, distress)

    x = base_amp * np.random.randn(num_samples) * (0.7 + 0.6 * env)
    y = base_amp * np.random.randn(num_samples) * (0.7 + 0.6 * env)
    z = 9.8 + base_amp * np.random.randn(num_samples) * (0.3 + 0.5 * env)

    # Occasional jump/fall during distress
    if distress and np.random.rand() < 0.3:
        idx = np.random.randint(0, num_samples)
        length = min(num_samples - idx, int(sr * 0.2))
        z[idx:idx + length] += np.linspace(3.0, 0.0, length)

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "z": z.tolist(),
        "sampling_rate": sr,
        "duration_sec": duration,
        "label": "distress" if distress else "normal",
    }


# ---------------------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Generate synthetic biometric data (children 3–12 yrs)"
    )
    parser.add_argument("--n-distress", type=int, default=10)
    parser.add_argument("--n-normal", type=int, default=10)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--age-years", type=float, default=7.0)
    parser.add_argument("--activity-level", type=float, default=0.5)
    parser.add_argument("--hrv-dir", type=str, default="data/synthetic/biometric/hrv")
    parser.add_argument("--accel-dir", type=str, default="data/synthetic/biometric/accelerometer")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    hrv_dir = Path(args.hrv_dir)
    accel_dir = Path(args.accel_dir)
    hrv_dir.mkdir(parents=True, exist_ok=True)
    accel_dir.mkdir(parents=True, exist_ok=True)

    def attach_metadata(data: dict, label: str, idx: int) -> dict:
        out = data.copy()
        out["sample_id"] = f"{label}_{idx:03d}"
        out["timestamp"] = datetime.utcnow().isoformat() + "Z"
        out["generator_version"] = "v2-children-3to12"
        out["child_age_years"] = float(np.clip(args.age_years, 3.0, 12.0))
        out["activity_level"] = args.activity_level
        return out

    print("Generating synthetic biometric data (children 3–12 years)...")

    # Distress samples
    for i in range(args.n_distress):
        hrv = attach_metadata(
            generate_distress_hrv(args.duration, args.age_years, args.activity_level),
            "distress",
            i,
        )
        with open(hrv_dir / f"distress_{i:03d}.json", "w") as f:
            json.dump(hrv, f, indent=2)

        acc = attach_metadata(
            generate_accelerometer_data(args.duration, True, args.age_years, args.activity_level),
            "distress",
            i,
        )
        with open(accel_dir / f"distress_{i:03d}.json", "w") as f:
            json.dump(acc, f, indent=2)

    # Normal samples
    for i in range(args.n_normal):
        hrv = attach_metadata(
            generate_normal_hrv(args.duration, args.age_years, args.activity_level),
            "normal",
            i,
        )
        with open(hrv_dir / f"normal_{i:03d}.json", "w") as f:
            json.dump(hrv, f, indent=2)

        acc = attach_metadata(
            generate_accelerometer_data(args.duration, False, args.age_years, args.activity_level),
            "normal",
            i,
        )
        with open(accel_dir / f"normal_{i:03d}.json", "w") as f:
            json.dump(acc, f, indent=2)

    print("Synthetic biometric data generation complete.")
    print(f"HRV directory: {hrv_dir}")
    print(f"Accelerometer directory: {accel_dir}")
