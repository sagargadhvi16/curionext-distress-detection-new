import numpy as np
import pytest

from src.biometric.augmentation import BiometricAugmentation


def test_augmentation_initialization():
    aug = BiometricAugmentation()
    assert aug is not None


def test_jitter_preserves_shape():
    aug = BiometricAugmentation()
    x = np.random.randn(100, 10)

    y = aug.jitter(x)

    assert y.shape == x.shape
    assert not np.allclose(x, y)  # should change values


def test_scaling_preserves_shape():
    aug = BiometricAugmentation()
    x = np.random.randn(80, 6)

    y = aug.scaling(x)

    assert y.shape == x.shape


def test_time_warp_preserves_shape():
    aug = BiometricAugmentation()
    x = np.random.randn(120, 8)

    y = aug.time_warp(x)

    assert y.shape == x.shape


def test_full_augmentation_pipeline():
    aug = BiometricAugmentation()
    x = np.random.randn(60, 12)

    y = aug.augment(x)

    assert y.shape == x.shape


def test_no_nan_or_inf():
    aug = BiometricAugmentation()
    x = np.random.randn(100, 5)

    y = aug.augment(x)

    assert np.isfinite(y).all()
