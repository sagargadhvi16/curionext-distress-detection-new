"""HRV feature extraction."""

import numpy as np
from typing import Dict
from scipy import signal
import warnings


def extract_hrv_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Extract time and frequency domain HRV features.
    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)

    if len(rr_intervals) < 3:
        return {
            "RMSSD": 0.0,
            "SDNN": 0.0,
            "pNN50": 0.0,
            "LF": 0.0,
            "HF": 0.0,
            "LF_HF": 0.0,
        }

    time_features = compute_time_domain_features(rr_intervals)

    if len(rr_intervals) >= 8:
        freq_features = compute_frequency_domain_features(rr_intervals)
    else:
        freq_features = {"LF": 0.0, "HF": 0.0, "LF_HF": 0.0}

    return {**time_features, **freq_features}


def compute_time_domain_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute time-domain HRV features.
    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)

    if len(rr_intervals) < 3:
        return {"RMSSD": 0.0, "SDNN": 0.0, "pNN50": 0.0}

    # Convert ms → seconds if needed
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


def compute_frequency_domain_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute frequency-domain HRV features using Lomb–Scargle.
    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)

    try:
        rr_sec = rr_intervals / 1000.0 if np.mean(rr_intervals) > 100 else rr_intervals

        time_points = np.cumsum(rr_sec)
        time_points -= time_points[0]

        if time_points[-1] < 32:
            return {"LF": 0.0, "HF": 0.0, "LF_HF": 0.0}

        rr_detrended = rr_sec - np.mean(rr_sec)

        freqs = np.linspace(0.04, 0.4, 200)
        power = signal.lombscargle(
            time_points, rr_detrended, freqs * 2 * np.pi
        )

        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)

        lf = np.trapz(power[lf_mask], freqs[lf_mask])
        hf = np.trapz(power[hf_mask], freqs[hf_mask])

        return {
            "LF": float(lf),
            "HF": float(hf),
            "LF_HF": float(lf / hf if hf > 0 else 0.0),
        }

    except Exception as e:
        warnings.warn(f"Frequency HRV failed: {e}")
        return {"LF": 0.0, "HF": 0.0, "LF_HF": 0.0}


def compute_frequency_domain_features_fft(
    rr_intervals: np.ndarray,
    fs: float = 4.0
) -> Dict[str, float]:
    """
    Compute HRV frequency-domain features using FFT.

    Bands:
    - VLF: 0.003–0.04 Hz
    - LF:  0.04–0.15 Hz
    - HF:  0.15–0.40 Hz
    """

    features = {
        "VLF": 0.0,
        "LF": 0.0,
        "HF": 0.0,
        "LF_HF": 0.0,
    }

    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)

    if len(rr_intervals) < 8:
        return features

    rr_sec = rr_intervals / 1000.0 if np.mean(rr_intervals) > 100 else rr_intervals

    if np.any(rr_sec <= 0):
        return features

    time_axis = np.cumsum(rr_sec)
    duration = time_axis[-1]

    if duration < 10.0:
        return features

    # Interpolate to uniform sampling
    t_uniform = np.arange(0, duration, 1 / fs)
    rr_interp = np.interp(t_uniform, time_axis, rr_sec)

    rr_interp -= np.mean(rr_interp)

    freqs = np.fft.rfftfreq(len(rr_interp), d=1 / fs)
    power = np.abs(np.fft.rfft(rr_interp)) ** 2

    def band_power(low, high):
        mask = (freqs >= low) & (freqs < high)
        return np.trapz(power[mask], freqs[mask]) if np.any(mask) else 0.0

    vlf = band_power(0.003, 0.04)
    lf = band_power(0.04, 0.15)
    hf = band_power(0.15, 0.40)

    features["VLF"] = float(vlf)
    features["LF"] = float(lf)
    features["HF"] = float(hf)
    features["LF_HF"] = float(lf / hf) if hf > 0 else 0.0

    return features


def compute_nonlinear_hrv_features(
    rr_intervals: np.ndarray,
    m: int = 2,
    r_ratio: float = 0.2
) -> Dict[str, float]:
    """
    Compute nonlinear HRV features:
    - Sample Entropy (SampEn)
    - Poincaré plot features (SD1, SD2)

    Args:
        rr_intervals: RR intervals (ms or sec)
        m: embedding dimension for SampEn
        r_ratio: tolerance as fraction of std

    Returns:
        Dictionary with SAMPEN, SD1, SD2
    """

    features = {
        "SAMPEN": 0.0,
        "SD1": 0.0,
        "SD2": 0.0
    }

    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)

    if len(rr_intervals) < 10:
        return features

    # ms → sec if needed
    rr_sec = rr_intervals / 1000.0 if np.mean(rr_intervals) > 100 else rr_intervals

    if np.std(rr_sec) == 0:
        return features

    # ---------- Sample Entropy ----------
    def sample_entropy(signal, m, r):
        n = len(signal)

        def _count_similar(template_len):
            count = 0
            for i in range(n - template_len):
                template = signal[i:i + template_len]
                for j in range(i + 1, n - template_len):
                    if np.all(np.abs(template - signal[j:j + template_len]) <= r):
                        count += 1
            return count

        r = r_ratio * np.std(signal)
        count_m = _count_similar(m)
        count_m1 = _count_similar(m + 1)

        if count_m == 0 or count_m1 == 0:
            return 0.0

        return -np.log(count_m1 / count_m)

    try:
        sampen = sample_entropy(rr_sec, m, r_ratio)
    except Exception:
        sampen = 0.0

    # ---------- Poincaré Plot Features ----------
    diff_rr = np.diff(rr_sec)

    sd1 = np.sqrt(np.var(diff_rr) / 2.0)
    sd2 = np.sqrt(2 * np.var(rr_sec) - (np.var(diff_rr) / 2.0))

    features["SAMPEN"] = float(sampen)
    features["SD1"] = float(sd1 * 1000)  # back to ms
    features["SD2"] = float(sd2 * 1000)

    return features


def extract_all_hrv_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Aggregate all HRV features into a single dictionary.

    Includes:
    - Time-domain HRV
    - Frequency-domain HRV (Lomb–Scargle)
    - Frequency-domain HRV (FFT)
    - Nonlinear HRV features
    """

    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)

    features = {}

    # -------- Time-domain --------
    try:
        features.update(compute_time_domain_features(rr_intervals))
    except Exception:
        features.update({"RMSSD": 0.0, "SDNN": 0.0, "pNN50": 0.0})

    # -------- Frequency-domain (Lomb–Scargle) --------
    try:
        features.update(compute_frequency_domain_features(rr_intervals))
    except Exception:
        features.update({"LF": 0.0, "HF": 0.0, "LF_HF": 0.0})

    # -------- Frequency-domain (FFT) --------
    try:
        fft_features = compute_frequency_domain_features_fft(rr_intervals)

        # Avoid overwriting LF/HF from Lomb–Scargle
        fft_features = {
            "FFT_" + k: v for k, v in fft_features.items()
        }

        features.update(fft_features)
    except Exception:
        features.update({
            "FFT_VLF": 0.0,
            "FFT_LF": 0.0,
            "FFT_HF": 0.0,
            "FFT_LF_HF": 0.0,
        })

    # -------- Nonlinear HRV --------
    try:
        features.update(compute_nonlinear_hrv_features(rr_intervals))
    except Exception:
        features.update({"SAMPEN": 0.0, "SD1": 0.0, "SD2": 0.0})

    return features