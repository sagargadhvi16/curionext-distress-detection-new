"""HRV feature extraction and aggregation."""

import numpy as np
from typing import Dict
from scipy import signal
import warnings


# ============================================================
# RR INTERVAL EXTRACTION
# ============================================================

def extract_rr_intervals(signal_data: np.ndarray, sr: float):
    """
    Extract RR intervals from raw ECG or PPG signal using peak detection.
    """
    quality = {
        "signal_ok": False,
        "reason": "",
        "num_beats": 0
    }

    if signal_data is None or len(signal_data) < sr * 2:
        quality["reason"] = "Signal too short"
        return np.array([]), quality

    signal_data = np.asarray(signal_data, dtype=np.float64)

    signal_data -= np.mean(signal_data)
    std = np.std(signal_data)
    if std == 0:
        quality["reason"] = "Flat signal"
        return np.array([]), quality
    signal_data /= std

    min_distance = int(0.3 * sr)  # ~200 BPM upper bound

    peaks, _ = signal.find_peaks(
        signal_data,
        distance=min_distance,
        prominence=0.5
    )

    if len(peaks) < 3:
        quality["reason"] = "Insufficient peaks detected"
        return np.array([]), quality

    rr_intervals_sec = np.diff(peaks) / sr
    rr_intervals_ms = rr_intervals_sec * 1000.0

    rr_intervals_ms = rr_intervals_ms[
        (rr_intervals_ms > 300) & (rr_intervals_ms < 2000)
    ]

    if len(rr_intervals_ms) < 3:
        quality["reason"] = "Invalid RR intervals"
        return np.array([]), quality

    quality.update({
        "signal_ok": True,
        "num_beats": len(rr_intervals_ms),
        "mean_rr_ms": float(np.mean(rr_intervals_ms))
    })

    return rr_intervals_ms, quality


# ============================================================
# HRV TIME-DOMAIN FEATURES
# ============================================================

def extract_hrv_time_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    RMSSD, SDNN, pNN50, mean HR, HR range
    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)

    features = {
        "RMSSD": 0.0,
        "SDNN": 0.0,
        "pNN50": 0.0,
        "mean_HR": 0.0,
        "HR_range": 0.0,
    }

    if len(rr_intervals) < 3:
        return features

    features.update(compute_time_domain_features(rr_intervals))

    rr_sec = rr_intervals / 1000.0 if np.mean(rr_intervals) > 100 else rr_intervals
    hr = 60.0 / rr_sec
    hr = hr[(hr > 30) & (hr < 220)]

    if len(hr) > 0:
        features["mean_HR"] = float(np.mean(hr))
        features["HR_range"] = float(np.max(hr) - np.min(hr))

    return features


# ============================================================
# HRV FREQUENCY-DOMAIN FEATURES
# ============================================================

def extract_hrv_freq_features(
    rr_intervals: np.ndarray,
    method: str = "welch"
) -> Dict[str, float]:
    """
    VLF, LF, HF power and LF/HF ratio
    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)

    features = {
        "VLF": 0.0,
        "LF": 0.0,
        "HF": 0.0,
        "LF_HF": 0.0,
    }

    if len(rr_intervals) < 8:
        return features

    features.update(compute_frequency_domain_features_fft(rr_intervals))
    return features


# ============================================================
# HRV NONLINEAR FEATURES
# ============================================================

def extract_hrv_nonlinear_features(
    rr_intervals: np.ndarray
) -> Dict[str, float]:
    """
    Sample Entropy, Poincaré SD1 / SD2
    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)

    features = {
        "SAMPEN": 0.0,
        "SD1": 0.0,
        "SD2": 0.0,
    }

    if len(rr_intervals) < 10:
        return features

    features.update(compute_nonlinear_hrv_features(rr_intervals))
    return features


# ============================================================
# HRV FEATURE AGGREGATOR (DELIVERABLE)
# ============================================================

class HRVFeatureExtractor:
    """
    Unified HRV feature extractor.
    """

    def __init__(self, freq_method: str = "welch"):
        self.freq_method = freq_method

    def extract_all(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        rr_intervals = np.asarray(rr_intervals, dtype=np.float64)

        features: Dict[str, float] = {}

        features.update(extract_hrv_time_features(rr_intervals))
        features.update(
            extract_hrv_freq_features(
                rr_intervals, method=self.freq_method
            )
        )
        features.update(extract_hrv_nonlinear_features(rr_intervals))

        return features


# ============================================================
# CORE HRV COMPUTATION UTILITIES
# ============================================================

def compute_time_domain_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    if len(rr_intervals) < 3:
        return {"RMSSD": 0.0, "SDNN": 0.0, "pNN50": 0.0}

    rr_sec = rr_intervals / 1000.0 if np.mean(rr_intervals) > 100 else rr_intervals
    diff_rr = np.diff(rr_sec)

    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    sdnn = np.std(rr_sec, ddof=1)

    diff_rr_ms = np.abs(diff_rr) * 1000
    pnn50 = (np.sum(diff_rr_ms > 50) / len(diff_rr)) * 100

    return {
        "RMSSD": float(rmssd * 1000),
        "SDNN": float(sdnn * 1000),
        "pNN50": float(pnn50),
    }


def compute_frequency_domain_features_fft(
    rr_intervals: np.ndarray,
    fs: float = 4.0
) -> Dict[str, float]:

    features = {"VLF": 0.0, "LF": 0.0, "HF": 0.0, "LF_HF": 0.0}

    if len(rr_intervals) < 8:
        return features

    rr_sec = rr_intervals / 1000.0 if np.mean(rr_intervals) > 100 else rr_intervals
    time_axis = np.cumsum(rr_sec)
    duration = time_axis[-1]

    if duration < 10:
        return features

    t_uniform = np.arange(0, duration, 1 / fs)
    rr_interp = np.interp(t_uniform, time_axis, rr_sec)
    rr_interp -= np.mean(rr_interp)

    freqs = np.fft.rfftfreq(len(rr_interp), d=1 / fs)
    power = np.abs(np.fft.rfft(rr_interp)) ** 2

    def band(low, high):
        m = (freqs >= low) & (freqs < high)
        return np.trapz(power[m], freqs[m]) if np.any(m) else 0.0

    features["VLF"] = band(0.003, 0.04)
    features["LF"] = band(0.04, 0.15)
    features["HF"] = band(0.15, 0.40)
    features["LF_HF"] = features["LF"] / features["HF"] if features["HF"] > 0 else 0.0

    return features


def compute_nonlinear_hrv_features(
    rr_intervals: np.ndarray,
    m: int = 2,
    r_ratio: float = 0.2
) -> Dict[str, float]:

    features = {"SAMPEN": 0.0, "SD1": 0.0, "SD2": 0.0}

    if len(rr_intervals) < 10:
        return features

    rr_sec = rr_intervals / 1000.0 if np.mean(rr_intervals) > 100 else rr_intervals
    if np.std(rr_sec) == 0:
        return features

    def sample_entropy(signal, m, r):
        n = len(signal)

        def _count_similar(k):
            count = 0
            for i in range(n - k):
                for j in range(i + 1, n - k):
                    if np.all(np.abs(signal[i:i + k] - signal[j:j + k]) <= r):
                        count += 1
            return count

        r = r_ratio * np.std(signal)
        cm = _count_similar(m)
        cm1 = _count_similar(m + 1)

        if cm == 0 or cm1 == 0:
            return 0.0
        return -np.log(cm1 / cm)

    try:
        sampen = sample_entropy(rr_sec, m, r_ratio)
    except Exception:
        sampen = 0.0

    diff_rr = np.diff(rr_sec)
    sd1 = np.sqrt(np.var(diff_rr) / 2.0)
    sd2 = np.sqrt(2 * np.var(rr_sec) - (np.var(diff_rr) / 2.0))

    features["SAMPEN"] = float(sampen)
    features["SD1"] = float(sd1 * 1000)
    features["SD2"] = float(sd2 * 1000)

    return features
