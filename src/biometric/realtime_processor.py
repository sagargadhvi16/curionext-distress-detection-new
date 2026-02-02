"""
Real-Time Biometric Processing Module (Sliding Window)

Handles:
- Streaming accelerometer data (16 Hz)
- Streaming HRV (RR intervals)
- Sliding window feature extraction
"""

from collections import deque
from typing import Dict
import numpy as np

from src.biometric.biometric_aggregator import extract_biometric_features


class RealTimeBiometricProcessor:
    """
    Sliding window processor for real-time biometric data.
    """

    def __init__(
        self,
        accel_sampling_rate: int = 16,
        accel_window_sec: int = 5,
        max_rr_intervals: int = 60,
        min_rr_intervals: int = 10
    ):
        """
        Args:
            accel_sampling_rate: Hz (16 for your case)
            accel_window_sec: window length in seconds
            max_rr_intervals: max RR intervals to keep
            min_rr_intervals: minimum RR needed for HRV
        """

        self.accel_window_size = accel_sampling_rate * accel_window_sec
        self.min_rr_intervals = min_rr_intervals

        # Sliding buffers
        self.accel_buffer = deque(maxlen=self.accel_window_size)
        self.rr_buffer = deque(maxlen=max_rr_intervals)

    # -------------------------------------------------
    # Add incoming data
    # -------------------------------------------------
    def add_accel_sample(self, x: float, y: float, z: float) -> None:
        """
        Add one accelerometer sample.
        """
        self.accel_buffer.append([x, y, z])

    def add_rr_interval(self, rr: float) -> None:
        """
        Add one RR interval (ms).
        """
        self.rr_buffer.append(rr)

    # -------------------------------------------------
    # Check readiness
    # -------------------------------------------------
    def is_ready(self) -> bool:
        """
        Check if enough data is available to process.
        """
        accel_ready = len(self.accel_buffer) == self.accel_window_size
        rr_ready = len(self.rr_buffer) >= self.min_rr_intervals

        return accel_ready and rr_ready

    # -------------------------------------------------
    # Process window
    # -------------------------------------------------
    def process(self) -> Dict[str, object]:
        """
        Extract biometric features from current window.
        """

        if not self.is_ready():
            raise RuntimeError("Not enough data to process")

        accel_data = np.array(self.accel_buffer, dtype=np.float32)
        rr_intervals = np.array(self.rr_buffer, dtype=np.float32)

        features = extract_biometric_features(
            rr_intervals=rr_intervals,
            accel_data=accel_data
        )

        return features

    # -------------------------------------------------
    # Reset buffers (optional)
    # -------------------------------------------------
    def reset(self) -> None:
        """
        Clear buffers after processing.
        """
        self.accel_buffer.clear()
        self.rr_buffer.clear()
