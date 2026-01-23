"""
Sanity evaluation: Emotion2Vec + YAMNet fusion
- Variable-length audio
- Chunk-level embeddings
- Simple linear classifier (low-data regime)

Environment assumptions:
- Python 3.10+
- torch >= 2.0
- tensorflow >= 2.19
- funasr == 1.3.0
- modelscope installed
"""

import torch
import torch.nn as nn
import librosa
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from funasr import AutoModel

# =====================
# CONFIG
# =====================
SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 5
LR = 1e-3
CHUNK_SEC = 6

# label mapping
LABELS = {
    "cry": 0,
    "fear": 1,
    "anger": 2,
    "neutral": 3,
}

# =====================
# DATA (COLAB FILES)
# =====================
# Files must exist in /content/
TRAIN_FILES = [
    ("/content/cry_medium.wav", 0),
    ("/content/fear_medium.wav", 1),
    ("/content/verbal_medium.wav", 2),
    ("/content/esc_medium.wav", 3),
]

TEST_FILES = [
    ("/content/cry_short.wav", 0),
    ("/content/fear_short.wav", 1),
    ("/content/verbal_short.wav", 2),
    ("/content/esc_short.wav", 3),
]

# =====================
# AUDIO CHUNKING
# =====================
def chunk_audio(audio, sr, chunk_sec):
    chunk_len = int(sr * chunk_sec)

    if len(audio) <= chunk_len:
        return [audio]

    return [
        audio[i : i + chunk_len]
        for i in range(0, len(audio) - chunk_len + 1, chunk_len)
    ]

# =====================
# LOAD MODELS
# =====================
print("Loading Emotion2Vec (FunASR / ModelScope)...")
emo_model = AutoModel(
    model="iic/emotion2vec_base",
    model_revision="v2.0.4",
    trust_remote_code=True,
    disable_update=True,
    device=DEVICE,
)

print("Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

# projection layers (to keep fusion balanced)
emo_proj = nn.Linear(768, 256).to(DEVICE)
yam_proj = nn.Linear(1024, 256).to(DEVICE)

# =====================
# FEATURE EXTRACTION
# =====================
def emotion2vec_embedding(audio_np):
    """
    Input: 1D numpy audio
    Output: torch tensor [1, 768]
    """
    out = emo_model.generate(audio_np, sr=SR)
    feats = out[0]["feats"]          # (T, 768)
    emb = feats.mean(axis=0)         # (768,)
    return torch.from_numpy(emb).float().unsqueeze(0)

def yamnet_embedding(audio_np):
    """
    Input: 1D numpy audio
    Output: torch tensor [1, 1024]
    """
    audio_tf = tf.convert_to_tensor(audio_np, dtype=tf.float32)
    _, embeddings, _ = yamnet(audio_tf)
    emb = embeddings.numpy().mean(axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)

def extract_fused_embedding(audio):
    chunks = chunk_audio(audio, SR, CHUNK_SEC)
    fused_chunks = []

    for c in chunks:
        emo = emo_proj(emotion2vec_embedding(c).to(DEVICE))
        yam = yam_proj(yamnet_embedding(c).to(DEVICE))
        fused_chunks.append(torch.cat([emo, yam], dim=-1))  # (1, 512)

    return torch.mean(torch.stack(fused_chunks), dim=0)     # (1, 512)

# =====================
# CLASSIFIER
# =====================
class EmotionClassifier(nn.Module):
    """
    Shallow classifier – intentional.
    Small data + frozen encoders.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# =====================
# TRAIN + TEST
# =====================
def run():
    model = EmotionClassifier(len(LABELS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print("\n===== TRAIN =====")
    for epoch in range(EPOCHS):
        total_loss = 0.0

        for path, label in TRAIN_FILES:
            audio, _ = librosa.load(path, sr=SR)
            feat = extract_fused_embedding(audio)

            logits = model(feat)
            target = torch.tensor([label], device=DEVICE)

            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    print("\n===== TEST =====")
    correct = 0

    for path, label in TEST_FILES:
        audio, _ = librosa.load(path, sr=SR)
        feat = extract_fused_embedding(audio)

        with torch.no_grad():
            pred = model(feat).argmax(dim=1).item()

        print(f"{path} → pred={pred}, gt={label}")
        correct += int(pred == label)

    print(f"\nAccuracy: {correct}/{len(TEST_FILES)}")

# =====================
if __name__ == "__main__":
    run()
