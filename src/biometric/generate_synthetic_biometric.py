import numpy as np
import pandas as pd
import random
from tqdm import tqdm

from hrv import HRVFeatureExtractor
from accelerometer import extract_accelerometer_features

# ================================
# GLOBAL CONFIG
# ================================
ACCEL_SR = 16
WINDOW_SEC = 60
N_SAMPLES = 1200
AMBIGUITY_PROB = 0.10

# Age buckets derived from pediatric vitals table
AGE_BUCKETS = {
    "3_5": {
        "hr_normal": (80, 130),
        "hr_stress": (110, 150),
        "hr_panic": (130, 170),
        "hrv_std": 70
    },
    "6_9": {
        "hr_normal": (75, 120),
        "hr_stress": (100, 140),
        "hr_panic": (120, 160),
        "hrv_std": 60
    },
    "10_12": {
        "hr_normal": (65, 110),
        "hr_stress": (90, 130),
        "hr_panic": (110, 150),
        "hrv_std": 50
    }
}

STATES = ["normal", "stress", "panic", "fall"]
ACTIVITIES = ["sedentary", "light", "moderate", "vigorous"]

# ================================
# RR INTERVAL GENERATION (CLINICAL)
# ================================
def generate_rr_intervals(state, age_bucket, duration_sec=60):

    cfg = AGE_BUCKETS[age_bucket]

    if state == "normal":
        hr_low, hr_high = cfg["hr_normal"]
    elif state == "stress":
        hr_low, hr_high = cfg["hr_stress"]
    else:  # panic / fall
        hr_low, hr_high = cfg["hr_panic"]

    mean_hr = np.random.uniform(hr_low, hr_high)
    mean_rr = 60000 / mean_hr
    std_rr = cfg["hrv_std"]

    rr, total = [], 0
    while total < duration_sec * 1000:
        val = np.clip(np.random.normal(mean_rr, std_rr), 300, 1200)
        rr.append(val)
        total += val

    return np.array(rr)

# ================================
# ACCELEROMETER GENERATION
# ================================
def generate_accelerometer(activity, state, age_bucket, duration_sec=60, sr=16):
    n = duration_sec * sr
    t = np.linspace(0, duration_sec, n)

    amp_map = {"sedentary": 0.02, "light": 0.1, "moderate": 0.3, "vigorous": 0.6}
    noise_map = {"sedentary": 0.02, "light": 0.05, "moderate": 0.1, "vigorous": 0.2}

    age_noise_factor = {
        "3_5": 1.3,
        "6_9": 1.1,
        "10_12": 0.9
    }[age_bucket]

    amp = amp_map[activity]
    noise = noise_map[activity] * age_noise_factor

    x = amp * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, noise, n)
    y = amp * np.sin(2 * np.pi * 2 * t + 1) + np.random.normal(0, noise, n)
    z = 1 + np.random.normal(0, noise, n)

    if state == "fall":
        idx = random.randint(int(0.3 * n), int(0.6 * n))
        x[idx:idx+3] += np.random.uniform(3, 5)
        y[idx:idx+3] += np.random.uniform(3, 5)
        z[idx:idx+3] += np.random.uniform(3, 5)

        x[idx+5:] = np.random.normal(0, 0.02, n - idx - 5)
        y[idx+5:] = np.random.normal(0, 0.02, n - idx - 5)
        z[idx+5:] = np.random.normal(1.0, 0.02, n - idx - 5)

    return np.stack([x, y, z], axis=1)

# ================================
# FEATURE NOISE
# ================================
def add_feature_noise(features, noise_level=0.30):
    noisy = {}
    for k, v in features.items():
        if isinstance(v, (int, float)) and not np.isnan(v):
            noisy[k] = v + np.random.normal(0, noise_level * (abs(v) + 1e-3))
        else:
            noisy[k] = v
    return noisy

# ================================
# SINGLE SAMPLE
# ================================
def generate_one_sample():

    age_bucket = random.choice(list(AGE_BUCKETS.keys()))
    state = random.choices(STATES, weights=[0.55, 0.25, 0.15, 0.05])[0]
    activity = random.choice(ACTIVITIES)

    rr = generate_rr_intervals(state, age_bucket, WINDOW_SEC)
    accel = generate_accelerometer(activity, state, age_bucket, WINDOW_SEC, ACCEL_SR)

    hrv_features = HRVFeatureExtractor().extract_all(rr)
    accel_features = extract_accelerometer_features(accel, ACCEL_SR)

    features = {**hrv_features, **accel_features}

    label = 1 if state in ["stress", "panic", "fall"] else 0
    is_ambiguous = 0

    if np.random.rand() < AMBIGUITY_PROB:
        features = add_feature_noise(features)
        label = 1 - label
        is_ambiguous = 1

    features["anomaly"] = label
    features["is_ambiguous"] = is_ambiguous
    features["age_bucket"] = age_bucket
    features["state_label"] = state
    features["activity_label"] = activity

    return features

# ================================
# DATASET GENERATION
# ================================
def generate_dataset(n_samples=1200):
    return pd.DataFrame([generate_one_sample() for _ in tqdm(range(n_samples))])

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    df = generate_dataset(N_SAMPLES)
    df.to_csv("synthetic_biometric_dataset.csv", index=False)

    print("✅ Clinically aligned child biometric dataset generated")
    print("Samples:", len(df))
    print("\nAge bucket distribution:")
    print(df["age_bucket"].value_counts())
    print("\nAnomaly distribution:")
    print(df["anomaly"].value_counts())
