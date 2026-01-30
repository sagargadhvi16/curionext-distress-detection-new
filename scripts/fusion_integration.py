"""
Fusion Integration Script for CurioNext
"""
import numpy as np
from src.fusion.signal_transformation import add_noise_with_SNR, GenerateRandomCurves, MagWarp


def integrate_fusion(sample):
    """Integrate fusion techniques into the distress detection pipeline."""
    noised = add_noise_with_SNR(sample, noise_amount=15)
    random_curves = GenerateRandomCurves(sample)
    # Further integration logic can be added here
    return noised, random_curves

# Example usage:
if __name__ == '__main__':
    sample = np.random.rand(16000)  # Example sample
    noised_sample, curves = integrate_fusion(sample)
    print("Noised Sample:", noised_sample)
    print("Random Curves:", curves)