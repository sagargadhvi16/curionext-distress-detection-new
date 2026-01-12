"""
RR Interval Extraction (Wearable ECG )
Pan–Tompkins–style pipeline (lightweight)
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# ---------------------------------
# Constants
# ---------------------------------
MIN_SIGNAL_DURATION_SEC = 5
RR_MS_MIN = 300    # 200 bpm
RR_MS_MAX = 2000   # 30 bpm

# ---------------------------------
# RR extraction
# ---------------------------------
def extract_rr_intervals_ecg(
    signal: np.ndarray,
    sr: float = 100.0
) -> np.ndarray | None:
    """
    Extract RR intervals .

    Args:
        signal: Raw signal
        sr: Sampling rate (default 50Hz)

    Returns:
        RR intervals in milliseconds, or None if invalid
    """
    signal = np.asarray(signal, dtype=float)

    if len(signal) < sr * MIN_SIGNAL_DURATION_SEC:
        return None

    # Bandpass filter (QRS band)
    nyq = sr / 2
    b, a = butter(2, [5/nyq, 15/nyq], btype="bandpass")
    filtered = filtfilt(b, a, signal)

    # Pan–Tompkins steps
    diff = np.diff(filtered)
    squared = diff ** 2
    window = max(int(0.15 * sr), 3)
    integrated = np.convolve(squared, np.ones(window)/window, mode="same")

    peaks, _ = find_peaks(
        integrated,
        distance=int(0.25 * sr),
        prominence=np.std(integrated) * 0.15
    )

    if len(peaks) < 3:
        return None

    rr_ms = np.diff(peaks) * 1000.0 / sr
    rr_ms = rr_ms[(rr_ms >= RR_MS_MIN) & (rr_ms <= RR_MS_MAX)]

    return rr_ms if len(rr_ms) >= 12 else None
