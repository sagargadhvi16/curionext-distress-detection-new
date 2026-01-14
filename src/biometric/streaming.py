"""
Real-Time Biometric Streaming Processor
=======================================

Implements sliding-window based real-time processing for biometric data.

Responsibilities:
- Buffer incoming biometric samples
- Maintain sliding time window
- Periodically extract features
- Optionally run encoder and anomaly detection


"""

from collections import deque
from typing import Optional, Dict
import time
import numpy as np
import torch

from src.biometric import hrv, accelerometer
from src.biometric.encoder import BiometricEncoder
from src.biometric.anomaly_detection import detect_biometric_anomaly


# =====================================================
# 🔐 FIXED FEATURE ORDER 
# =====================================================

FEATURE_ORDER = [
    # HRV
    "RMSSD", "SDNN", "pNN50", "MeanHR", "HR_Range",
    "VLF", "LF", "HF", "LF_HF",
    "SAMPEN", "SD1", "SD2", "SD1_SD2",

    # Accelerometer
    "ACC_MEAN_MAG", "ACC_STD_MAG", "ACC_MAX_MAG",
    "ACC_ENERGY", "ACC_PEAK_COUNT", "ACC_ORIENT_VAR",
]


class StreamingBiometricProcessor:
    """
    Real-time biometric processor using sliding windows.

    Design goals:
    - Deterministic behavior
    - No hidden state beyond window
    - Safe under missing or delayed data
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        step_seconds: float = 5.0,
        min_rr_count: int = 20,
        min_accel_samples: int = 50,
        encoder: Optional[BiometricEncoder] = None
    ):
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.min_rr_count = min_rr_count
        self.min_accel_samples = min_accel_samples

        # Buffers store (timestamp, value)
        self.rr_buffer = deque()
        self.accel_buffer = deque()

        self.last_emit_time: Optional[float] = None
        self.encoder = encoder

    # =====================================================
    # BUFFER INGESTION
    # =====================================================

    def add_rr_interval(
        self,
        rr_seconds: float,
        timestamp: Optional[float] = None
    ):
        """
        Add a single RR interval.

        IMPORTANT:
        - Input RR is in SECONDS
        - Internally stored as MILLISECONDS
        """
        ts = timestamp if timestamp is not None else time.time()
        rr_ms = float(rr_seconds) * 1000.0
        self.rr_buffer.append((ts, rr_ms))

    def add_accelerometer_sample(
        self,
        x: float,
        y: float,
        z: float,
        timestamp: Optional[float] = None
    ):
        """Add a single accelerometer sample."""
        ts = timestamp if timestamp is not None else time.time()
        self.accel_buffer.append((ts, np.array([x, y, z], dtype=float)))

    # =====================================================
    # INTERNAL BUFFER MAINTENANCE
    # =====================================================

    def _trim_buffer(self, buffer: deque):
        """
        Trim buffer based on latest sample timestamp
        (not wall-clock time).
        """
        if not buffer:
            return

        latest_ts = buffer[-1][0]

        while buffer and (latest_ts - buffer[0][0]) > self.window_seconds:
            buffer.popleft()

    # =====================================================
    # CORE PROCESSING
    # =====================================================

    def process(self) -> Optional[Dict]:
        """
        Run feature extraction if window + step conditions are met.

        Returns:
            dict with biometric features OR None
        """
        now = time.time()

        # Enforce step interval
        if self.last_emit_time is not None:
            if (now - self.last_emit_time) < self.step_seconds:
                return None

        # Trim buffers
        self._trim_buffer(self.rr_buffer)
        self._trim_buffer(self.accel_buffer)

        rr_values = [v for _, v in self.rr_buffer]              # ms
        accel_values = np.array([v for _, v in self.accel_buffer])

        # Data sufficiency checks
        if len(rr_values) < self.min_rr_count:
            return None

        if len(accel_values) < self.min_accel_samples:
            return None

        # Feature extraction
        try:
            hrv_features = hrv.extract_all_hrv_features(rr_values)
        except Exception:
            return None

        try:
            accel_features = accelerometer.extract_accelerometer_features(accel_values)
        except Exception:
            return None

        features = {**hrv_features, **accel_features}

        self.last_emit_time = now
        return features

    # =====================================================
    # OPTIONAL: ENCODING + ANOMALY
    # =====================================================

    def process_and_infer(self) -> Optional[Dict]:
        """
        Run feature extraction, encoder, and anomaly detection.

        Returns:
            dict with features, embedding, anomaly result OR None
        """
        features = self.process()
        if features is None:
            return None

        result = {"features": features}

        # -----------------------------
        # Optional encoding
        # -----------------------------
        if self.encoder is not None:
            feature_vector = np.array(
                [features.get(k, 0.0) for k in FEATURE_ORDER],
                dtype=float
            )

            feature_vector = (
                torch.tensor(feature_vector)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)   # (B=1, T=1, F)
            )

            with torch.no_grad():
                embedding = self.encoder(feature_vector)

            result["embedding"] = embedding.squeeze(0)

        # -----------------------------
        # Optional anomaly detection
        # -----------------------------
        try:
            result["anomaly"] = detect_biometric_anomaly(features)
        except Exception:
            result["anomaly"] = None

        return result

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        """Clear all buffers and timing state."""
        self.rr_buffer.clear()
        self.accel_buffer.clear()
        self.last_emit_time = None


# =====================================================
# 🧪 LOCAL SANITY TEST
# =====================================================

if __name__ == "__main__":
    processor = StreamingBiometricProcessor(
        step_seconds=0,
        min_rr_count=10,
        min_accel_samples=20
    )

    # Simulate RR intervals (~75 BPM → 0.8s)
    for _ in range(20):
        processor.add_rr_interval(0.8)

    # Simulate accelerometer samples
    for _ in range(100):
        processor.add_accelerometer_sample(0.1, 0.2, 9.7)

    output = processor.process()
    print("Features available:", output is not None)

    processor.reset()
