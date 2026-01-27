import torch
from modelscope import AutoModel

class Emotion2VecEncoder:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = AutoModel.from_pretrained(
            "iic/emotion2vec_base",
            revision="v2.0.4",
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def encode(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        """
        wav_tensor: [1, T] @ 16kHz
        returns: [D]
        """
        with torch.no_grad():
            out = self.model(wav_tensor)

        if isinstance(out, dict):
            feats = out.get("last_hidden_state", list(out.values())[0])
        else:
            feats = out

        # [1, T, D] → [D]
        if feats.dim() == 3:
            feats = feats.mean(dim=1)

        return feats.squeeze(0)
