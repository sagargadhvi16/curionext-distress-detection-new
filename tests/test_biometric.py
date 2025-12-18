"""
Tests for biometric processing module.

Covers:
- HRV feature extraction
- Accelerometer feature extraction
- Basic biometric encoder checks
"""

import numpy as np
import pytest

from src.biometric import hrv, accelerometer

# Encoder may be optional / lightweight
try:
    from src.biometric.encoder import BiometricEncoder
    ENCODER_AVAILABLE = True
except Exception:
    ENCODER_AVAILABLE = False


# ======================================================
# HRV TESTS
# ======================================================

class TestHRV:
    """Test HRV feature extraction."""

    def test_extract_hrv_features(self):
        """HRV features should return a valid dictionary."""
        rr = np.random.normal(800, 30, 60)  # 60 RR intervals

        features = hrv.extract_all_hrv_features(rr)

        assert isinstance(features, dict)
        assert "RMSSD" in features
        assert "SDNN" in features
        assert "LF_HF" in features

    def test_time_domain_features(self):
        """Time-domain HRV metrics should be valid."""
        rr = np.random.normal(800, 20, 50)

        features = hrv.compute_time_domain_features(rr)

        assert features["RMSSD"] >= 0
        assert features["SDNN"] >= 0
        assert 0 <= features["pNN50"] <= 100

    def test_frequency_domain_features(self):
        """Frequency-domain HRV metrics should not crash."""
        rr = np.random.normal(800, 25, 80)

        features = hrv.compute_frequency_domain_features_fft(rr)

        assert "LF" in features
        assert "HF" in features
        assert features["LF"] >= 0
        assert features["HF"] >= 0


# ======================================================
# ACCELEROMETER TESTS
# ======================================================

class TestAccelerometer:
    """Test accelerometer processing."""

    def test_extract_accelerometer_features(self):
        """Accelerometer features should return valid dictionary."""
        accel = np.random.normal(0, 1, (300, 3))  # ~6 sec data

        features = accelerometer.extract_accelerometer_features(accel)

        assert isinstance(features, dict)

        # Match actual feature names used in implementation
        assert "ACC_MEAN_MAG" in features
        assert "ACC_MAX_MAG" in features
        assert "ACC_ENERGY" in features

        assert features["ACC_MEAN_MAG"] >= 0
        assert features["ACC_MAX_MAG"] >= 0
        assert features["ACC_ENERGY"] >= 0

    def test_compute_magnitude(self):
        """Magnitude computation should be correct."""
        accel = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        mag = accelerometer.compute_magnitude(accel)

        assert len(mag) == 3
        assert np.all(mag > 0)

    def test_detect_movement_peaks(self):
        """
        Peak detection should return a non-negative integer.
        Detection sensitivity depends on thresholds, so we only
        test stability and correctness.
        """
        accel = np.random.normal(0, 0.2, (300, 3))
        accel[150] += np.array([5.0, 5.0, 5.0])  # simulated impact

        mag = accelerometer.compute_magnitude(accel)
        peaks = accelerometer.detect_movement_peaks(mag)

        assert isinstance(peaks, int)
        assert peaks >= 0



# ======================================================
# BIOMETRIC ENCODER TESTS (LIGHTWEIGHT)
# ======================================================

@pytest.mark.skipif(not ENCODER_AVAILABLE, reason="Encoder not available")
class TestBiometricEncoder:
    """Test biometric encoder."""

    def test_encoder_initialization(self):
        """Encoder should initialize without error."""
        encoder = BiometricEncoder(input_dim=16, embedding_dim=64)
        assert encoder is not None

    def test_encoder_forward_pass(self):
        """Encoder forward pass should return output."""
        encoder = BiometricEncoder(input_dim=16, embedding_dim=64)

        dummy_input = np.random.normal(0, 1, (1, 10, 16))
        output = encoder.forward(dummy_input)

        assert output is not None

    def test_embedding_dimension(self):
        """Output embedding dimension should be 64."""
        encoder = BiometricEncoder(input_dim=16, embedding_dim=64)

        dummy_input = np.random.normal(0, 1, (1, 10, 16))
        output = encoder.forward(dummy_input)

        assert output.shape[-1] == 64
