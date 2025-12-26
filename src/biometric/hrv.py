"""
HRV Feature Extraction Module
=============================

This module computes Heart Rate Variability (HRV) features from RR intervals.

Design principles:
- Single, explicit RR unit handling (SECONDS internally)
- Safe defaults for short windows
- Normalized spectral power
- Minimal hidden assumptions
- Test-friendly public API

References:
- Task Force of the European Society of Cardiology (1996)
- Shaffer & Ginsberg, Frontiers in Public Health
"""

import numpy as np
from scipy import signal

# =====================================================
# EXPLICIT CONSTANTS & ASSUMPTIONS
# =====================================================

RR_UNIT_INTERNAL = "seconds"     # all RR processed in seconds
MIN_RR_COUNT = 20               # minimum RR samples for HRV

# Frequency bands (Hz)
VLF_BAND = (0.003, 0.04)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)

# Lomb–Scargle frequency grid
LOMB_FREQS = np.linspace(0.01, 0.5, 512)

# FFT interpolation rate (Hz)
FFT_FS = 4.0

# Minimum duration (seconds) for VLF reliability
MIN_VLF_DURATION = 300.0  # 5 minutes


# =====================================================
# RR PREPROCESSING
# =====================================================

def preprocess_rr(rr_intervals):
    """
    Convert RR intervals to seconds and validate input.

    Args:
        rr_intervals: array-like RR intervals (ms or s)

    Returns:
        rr (np.ndarray): RR intervals in seconds
    """
    rr = np.asarray(rr_intervals, dtype=float)

    # Detect milliseconds and convert
    if np.nanmean(rr) > 10.0:
        rr = rr / 1000.0

    rr = rr[~np.isnan(rr)]
    rr = rr[rr > 0]

    if len(rr) < MIN_RR_COUNT:
        raise ValueError("Not enough RR intervals for HRV computation")

    return rr


# =====================================================
# TIME-DOMAIN FEATURES
# =====================================================

def compute_time_domain_features(rr_intervals):
    """
    Compute time-domain HRV features.

    Outputs:
        RMSSD, SDNN in milliseconds
        pNN50 in percentage
    """
    rr = preprocess_rr(rr_intervals)
    diff_rr = np.diff(rr)

    return {
        "RMSSD": np.sqrt(np.mean(diff_rr ** 2)) * 1000.0,   # ms
        "SDNN": np.std(rr) * 1000.0,                        # ms
        "pNN50": np.mean(np.abs(diff_rr) > 0.05) * 100.0   # %
    }


# =====================================================
# FREQUENCY-DOMAIN (LOMB–SCARGLE)
# =====================================================

def compute_frequency_domain_features_lomb(rr_intervals):
    """
    Compute HRV frequency features using Lomb–Scargle PSD.

    Notes:
    - Handles uneven RR sampling
    - PSD normalized by variance
    - VLF disabled for short windows
    """
    rr = preprocess_rr(rr_intervals)

    t = np.cumsum(rr) - rr[0]
    duration = t[-1] - t[0]

    rr_detrended = rr - np.mean(rr)

    pxx = signal.lombscargle(
        t,
        rr_detrended,
        2 * np.pi * LOMB_FREQS
    )

    # Normalize power by signal variance
    var = np.var(rr_detrended)
    if var > 0:
        pxx = pxx / var

    def band_power(band):
        mask = (LOMB_FREQS >= band[0]) & (LOMB_FREQS <= band[1])
        return np.trapz(pxx[mask], LOMB_FREQS[mask]) if np.any(mask) else np.nan

    vlf = band_power(VLF_BAND) if duration >= MIN_VLF_DURATION else np.nan
    lf = band_power(LF_BAND)
    hf = band_power(HF_BAND)

    return {
        "VLF": vlf,
        "LF": lf,
        "HF": hf,
        "LF_HF": lf / hf if hf > 0 else np.nan
    }


# =====================================================
# FREQUENCY-DOMAIN (FFT / WELCH)
# =====================================================

def compute_frequency_domain_features_fft(rr_intervals):
    """
    Compute HRV frequency features using FFT + Welch PSD.

    Notes:
    - Interpolates RR to uniform grid
    - Uses Hann window to reduce leakage
    - Power normalized (density)
    """
    rr = preprocess_rr(rr_intervals)

    t = np.cumsum(rr)
    duration = t[-1] - t[0]

    # Interpolate RR to uniform sampling
    t_uniform = np.arange(t[0], t[-1], 1.0 / FFT_FS)
    rr_interp = np.interp(t_uniform, t, rr)

    rr_interp = rr_interp - np.mean(rr_interp)

    freqs, pxx = signal.welch(
        rr_interp,
        fs=FFT_FS,
        window="hann",
        nperseg=min(256, len(rr_interp)),
        scaling="density"
    )

    def band_power(band):
        mask = (freqs >= band[0]) & (freqs <= band[1])
        return np.trapz(pxx[mask], freqs[mask]) if np.any(mask) else np.nan

    vlf = band_power(VLF_BAND) if duration >= MIN_VLF_DURATION else np.nan
    lf = band_power(LF_BAND)
    hf = band_power(HF_BAND)

    return {
        "VLF": vlf,
        "LF": lf,
        "HF": hf,
        "LF_HF": lf / hf if hf > 0 else np.nan
    }


# =====================================================
# PUBLIC API (TEST-SAFE)
# =====================================================

def extract_all_hrv_features(rr_intervals):
    """
    Unified HRV feature extractor.

    Returns:
        Dictionary containing:
        - Time-domain features
        - Frequency-domain features
    """
    features = {}

    # Time-domain (always safe)
    features.update(compute_time_domain_features(rr_intervals))

    # Frequency-domain (prefer Lomb–Scargle)
    try:
        features.update(
            compute_frequency_domain_features_lomb(rr_intervals)
        )
    except Exception:
        features.update(
            compute_frequency_domain_features_fft(rr_intervals)
        )

    return features
