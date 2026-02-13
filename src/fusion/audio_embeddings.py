import torch
import torch.nn as nn
import tensorflow as tf
import tensorflow_hub as hub
from funasr import AutoModel
import numpy as np

SR = 16000
DEVICE = "cpu"


class FusionEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()

        self.emo_model = AutoModel(
            model="iic/emotion2vec_base",
            model_revision="v2.0.4",
            trust_remote_code=True,
            device=DEVICE,
            disable_update=True
        )

        self.yamnet = hub.KerasLayer(
            "https://tfhub.dev/google/yamnet/1",
            trainable=False
        )

        self.emo_proj = nn.Linear(768, 256)
        self.yam_proj = nn.Linear(1024, 256)

    def emotion2vec_embedding(self, audio):
        out = self.emo_model.generate(audio, sr=SR, disable_progress_bar=True)
        feats = out[0]["feats"]
        if feats.ndim > 1:
            feats = feats.mean(axis=0)
        return torch.tensor(feats).float()

    def yamnet_embedding(self, audio):
        audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, embeddings, _ = self.yamnet(audio_tf)
        return torch.from_numpy(embeddings.numpy().mean(axis=0)).float()

    def forward(self, audio):
        emo = self.emo_proj(self.emotion2vec_embedding(audio))
        yam = self.yam_proj(self.yamnet_embedding(audio))
        return torch.cat([emo, yam], dim=-1)
