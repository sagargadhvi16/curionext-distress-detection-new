"""
Integrated Biometric Encoder

Combines:
- Per-child baseline normalization
- BiLSTM + Attention encoder

Input:
- Sequence of biometric feature dictionaries

Output:
- Fixed-length biometric embedding
"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn

from src.biometric.baseline import ChildBaselineModel
from src.biometric.encoder import BiometricEncoder


class IntegratedBiometricEncoder(nn.Module):
    """
    Complete biometric encoder with:
    - Per-child baseline normalization
    - Temporal BiLSTM + Attention encoder
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64
    ):
        super().__init__()

        # ---------------------------------
        # Baseline model (non-trainable)
        # ---------------------------------
        self.baseline_model = ChildBaselineModel()

        # ---------------------------------
        # Temporal encoder (BiLSTM + Attention)
        # ---------------------------------
        self.encoder = BiometricEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim
        )

    # -------------------------------------------------
    # Forward pass
    # -------------------------------------------------
    def forward(
        self,
        child_id: str,
        raw_feature_dicts: Optional[List[Dict[str, float]]] = None,
        feature_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode biometric sequence.

        Either provide:
        - raw_feature_dicts (List of dicts)
        OR
        - feature_sequence (Tensor)

        Args:
            child_id: Child identifier
            raw_feature_dicts: List of biometric feature dicts
            feature_sequence: Tensor (B, T, F)

        Returns:
            embedding: Tensor (B, embedding_dim)
        """

        # ---------------------------------
        # Convert dicts → tensor if needed
        # ---------------------------------
        if feature_sequence is None:
            if raw_feature_dicts is None:
                raise ValueError(
                    "Either raw_feature_dicts or feature_sequence must be provided"
                )

            # Normalize features using baseline
            normalized = [
                self.baseline_model.normalize(child_id, f)
                for f in raw_feature_dicts
            ]

            # Dict → numpy array (T, F)
            feature_sequence = np.array(
                [list(f.values()) for f in normalized],
                dtype=np.float32
            )

            # Convert to tensor and add batch dim
            feature_sequence = torch.tensor(feature_sequence).unsqueeze(0)

        # ---------------------------------
        # Encode using BiLSTM + Attention
        # ---------------------------------
        embedding = self.encoder(feature_sequence)

        return embedding


# -------------------------------------------------
# 🧪 Local test
# -------------------------------------------------
if __name__ == "__main__":
    encoder = IntegratedBiometricEncoder(input_dim=10)

    # -------------------------
    # Step 1: Fit baseline
    # -------------------------
    baseline_data = [
        {"RMSSD": 60, "LF_HF": 1.2, "ACC_MEAN_MAG": 1.0,
         "x": 1, "y": 1, "z": 1, "a": 1, "b": 1, "c": 1, "d": 1},
        {"RMSSD": 62, "LF_HF": 1.1, "ACC_MEAN_MAG": 1.1,
         "x": 1, "y": 1, "z": 1, "a": 1, "b": 1, "c": 1, "d": 1},
    ]

    encoder.baseline_model.fit("child_01", baseline_data)

    # -------------------------
    # Step 2: Incoming data
    # -------------------------
    features = [
        {"RMSSD": 30, "LF_HF": 3.0, "ACC_MEAN_MAG": 1.4,
         "x": 1, "y": 1, "z": 1, "a": 1, "b": 1, "c": 1, "d": 1},
        {"RMSSD": 28, "LF_HF": 3.5, "ACC_MEAN_MAG": 1.6,
         "x": 1, "y": 1, "z": 1, "a": 1, "b": 1, "c": 1, "d": 1},
    ]

    emb = encoder(
        child_id="child_01",
        raw_feature_dicts=features
    )

    print("Embedding shape:", emb.shape)
