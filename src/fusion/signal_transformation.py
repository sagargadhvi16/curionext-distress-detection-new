# -*- coding: utf-8 -*-
"""
Signal transformation functions adapted from WER-SSL project.
"""
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline


def add_noise_with_SNR(signal, noise_amount):
    """ 
    Adding noise to the signal based on a specified SNR.
    """
    target_snr_db = noise_amount
    x_watts = signal ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    noised_signal = signal + noise_volts
    return noised_signal


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = np.arange(0, X.shape[0], (X.shape[0]-1)/(knot+1))
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2,))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx, yy)
    return cs_x(x_range)


def MagWarp(X, sigma=0.2):
    """Magnitude warping function."""
    # Implementation of magnitude warping
    pass

# Additional transformation functions can be added here.