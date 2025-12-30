"""
Audio SHAP explainer using GradientExplainer (fast & stable).
"""

import shap
import torch
from typing import Optional, List


class AudioExplainer:
    def __init__(
        self,
        model: torch.nn.Module,
        background_data: torch.Tensor,
        device: str = "cpu"
    ):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        self.background = background_data.to(device)

        # ✅ FAST explainer for PyTorch models
        self.explainer = shap.GradientExplainer(
            self.model,
            self.background
        )

    def explain(
        self,
        audio_features: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ):
        audio_features = audio_features.to(self.device)

        shap_values = self.explainer.shap_values(audio_features)

        if feature_names is None:
            feature_names = [
                f"feature_{i}" for i in range(audio_features.shape[1])
            ]

        return {
            "shap_values": shap_values,
            "feature_names": feature_names
        }
