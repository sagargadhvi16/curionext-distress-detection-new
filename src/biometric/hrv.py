


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
        "RMSSD": float(rmssd * 1000),  # back to ms
        "SDNN": float(sdnn * 1000),
        "pNN50": float(pnn50),
    }


def compute_frequency_domain_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute frequency-domain HRV features.
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
