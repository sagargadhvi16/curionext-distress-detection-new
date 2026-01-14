import numpy as np
from scipy.interpolate import interp1d


class BiometricAugmentation:
    """
    Biometric data augmentation:
    - Jittering
    - Scaling
    - Time warping

    Input shape: (T, F)
    Output shape: (T, F)
    """

    def __init__(
        self,
        jitter_sigma: float = 0.02,
        scaling_sigma: float = 0.1,
        max_time_warp: float = 0.2
    ):
        self.jitter_sigma = jitter_sigma
        self.scaling_sigma = scaling_sigma
        self.max_time_warp = max_time_warp

    # ==================================================
    # JITTERING
    # ==================================================
    def jitter(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.jitter_sigma, size=x.shape)
        return x + noise

    # ==================================================
    # SCALING
    # ==================================================
    def scaling(self, x: np.ndarray) -> np.ndarray:
        scale = np.random.normal(1.0, self.scaling_sigma, size=(1, x.shape[1]))
        return x * scale

    # ==================================================
    # TIME WARPING (FIXED)
    # ==================================================
    def time_warp(self, x: np.ndarray) -> np.ndarray:
        """
        Time warping using smooth random curve + interpolation.
        Preserves shape exactly.
        """
        T, F = x.shape

        # Original time steps
        orig_time = np.arange(T)

        # Generate smooth random warp
        warp_strength = np.random.uniform(1 - self.max_time_warp,
                                          1 + self.max_time_warp)

        warped_time = orig_time * warp_strength

        # Normalize warped time to [0, T-1]
        warped_time = (warped_time - warped_time.min()) / (
            warped_time.max() - warped_time.min() + 1e-8
        ) * (T - 1)

        warped = np.zeros_like(x)

        for f in range(F):
            interp_fn = interp1d(
                warped_time,
                x[:, f],
                kind="linear",
                fill_value="extrapolate"
            )
            warped[:, f] = interp_fn(orig_time)

        return warped

    # ==================================================
    # FULL PIPELINE
    # ==================================================
    def augment(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()

        if np.random.rand() < 0.8:
            out = self.jitter(out)

        if np.random.rand() < 0.8:
            out = self.scaling(out)

        if np.random.rand() < 0.5:
            out = self.time_warp(out)

        # Safety
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return out
