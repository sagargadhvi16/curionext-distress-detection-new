"""
Per-Child Baseline Modeling for Biometric Features

Purpose:
- Each child has a different normal (resting) biometric pattern
- We learn a baseline per child
- New biometric features are normalized relative to that baseline

This improves personalization and reduces false alarms.
"""

from typing import Dict, List
import numpy as np


class ChildBaselineModel:
    """
    Maintains per-child baseline statistics and performs normalization.
    """

    def __init__(self):
        """
        Initialize baseline storage.

        Structure:
        self.baselines = {
            child_id: {
                feature_name: {
                    "mean": float,
                    "std": float
                }
            }
        }
        """
        self.baselines: Dict[str, Dict[str, Dict[str, float]]] = {}

    # ---------------------------------------------------------
    # STEP 1: Fit baseline for a child (calm / resting data)
    # ---------------------------------------------------------
    def fit(
        self,
        child_id: str,
        baseline_feature_windows: List[Dict[str, float]]
    ) -> None:
        """
        Fit baseline statistics for a child.

        Args:
            child_id: Unique child identifier
            baseline_feature_windows: List of feature dictionaries
                (collected during calm / normal state)

        Example input:
        [
            {"RMSSD": 60, "SDNN": 40, "ACC_MEAN_MAG": 1.0},
            {"RMSSD": 62, "SDNN": 42, "ACC_MEAN_MAG": 1.1}
        ]
        """

        if len(baseline_feature_windows) == 0:
            raise ValueError("Baseline data is empty")

        feature_names = baseline_feature_windows[0].keys()
        baseline_stats = {}

        for feature in feature_names:
            values = np.array(
                [window[feature] for window in baseline_feature_windows],
                dtype=np.float64
            )

            mean = float(np.mean(values))
            std = float(np.std(values)) if np.std(values) > 0 else 1e-6

            baseline_stats[feature] = {
                "mean": mean,
                "std": std
            }

        self.baselines[child_id] = baseline_stats

    # ---------------------------------------------------------
    # STEP 2: Normalize new features using child baseline
    # ---------------------------------------------------------
    def normalize(
        self,
        child_id: str,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize features relative to child's baseline.

        Uses z-score normalization:
        (value - baseline_mean) / baseline_std

        Args:
            child_id: Unique child identifier
            features: Current biometric feature dictionary

        Returns:
            Normalized feature dictionary
        """

        if child_id not in self.baselines:
            # Baseline not available → return raw features
            return features

        normalized_features = {}
        baseline_stats = self.baselines[child_id]

        for feature, value in features.items():
            if feature in baseline_stats:
                mean = baseline_stats[feature]["mean"]
                std = baseline_stats[feature]["std"]

                normalized_features[feature] = (value - mean) / std
            else:
                # Feature not part of baseline → keep as is
                normalized_features[feature] = value

        return normalized_features

    # ---------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------
    def has_baseline(self, child_id: str) -> bool:
        """
        Check if baseline exists for child.
        """
        return child_id in self.baselines

    def reset_child(self, child_id: str) -> None:
        """
        Remove baseline for a child (e.g., sensor re-calibration).
        """
        if child_id in self.baselines:
            del self.baselines[child_id]
