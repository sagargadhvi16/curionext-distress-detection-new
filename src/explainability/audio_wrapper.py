"""
Audio-only wrapper for SHAP explainability.

Exposes a single-input forward pass (audio features only)
while keeping the core multimodal model untouched.
"""

import torch
import torch.nn as nn


class AudioOnlyWrapper(nn.Module):
    def __init__(self, full_model: nn.Module):
        super().__init__()
        self.audio_encoder = full_model.audio_encoder
        self.fusion = full_model.fusion
        self.classifier = full_model.classifier

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: (B, audio_feature_dim)

        Returns:
            distress probability (B,)
        """
        audio_emb = self.audio_encoder(audio_features)

        batch_size = audio_emb.size(0)
        device = audio_emb.device

        # Dummy placeholders for other modalities
        bio_emb = torch.zeros(batch_size, 256, device=device)
        ctx_emb = torch.zeros(batch_size, 64, device=device)

        fused = self.fusion(audio_emb, bio_emb, ctx_emb)
        outputs = self.classifier(fused)

        probs = torch.softmax(outputs["distress_logits"], dim=-1)
        return probs[:, 1].unsqueeze(1)   # shape: (B, 1)
