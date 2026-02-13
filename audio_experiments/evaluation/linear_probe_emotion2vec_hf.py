"""
Emotion2Vec (HuggingFace, NO fairseq) feasibility experiment.

- Uses emotion2vec HF repo with trust_remote_code
- Raw waveform input (no feature extractor)
- Long-audio chunking
- Mean pooling
- Linear probe
"""

import torch
import torch.nn as nn
import librosa
import numpy as np

from transformers import AutoModel

# ======================================================
# CONFIG
# ======================================================
SR = 16000
DEVICE = "cpu"
EPOCHS = 5
LR = 1e-3
CHUNK_SEC = 6

# ======================================================
# DATA
# ======================================================
TRAIN_FILES = [
    ("audio_experiments/input/cry/cry_long.wav", 1),
    ("audio_experiments/input/fear/fear_long.wav", 1),
    ("audio_experiments/input/verbal_aggression/verbal_long.wav", 1),
    ("audio_experiments/input/background_noise/esc50/esc_long.wav", 0),
]

TEST_FILES = [
    ("audio_experiments/input/cry/cry_medium.wav", 1),
    ("audio_experiments/input/background_noise/esc50/esc_medium.wav", 0),
]

# ======================================================
# AUDIO CHUNKING
# ======================================================
def chunk_audio(audio, sr, chunk_sec):
    chunk_len = sr * chunk_sec
    return [
        audio[i:i + chunk_len]
        for i in range(0, len(audio) - chunk_len, chunk_len)
    ]

# ======================================================
# LOAD EMOTION2VEC (HF, CORRECT WAY)
# ======================================================
print("Loading emotion2vec (HF, trust_remote_code)...")

emotion2vec = AutoModel.from_pretrained(
    "emotion2vec/emotion2vec_base",
    trust_remote_code=True
).to(DEVICE).eval()

for p in emotion2vec.parameters():
    p.requires_grad = False

# Emotion2vec base → 768-dim
emo_proj = nn.Linear(768, 256)

# ======================================================
# LINEAR PROBE
# ======================================================
class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# ======================================================
# FEATURE EXTRACTION
# ======================================================
def emotion2vec_chunk_embedding(chunk):
    audio_t = torch.tensor(chunk).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = emotion2vec(audio_t)
        # emotion2vec returns [1, T, 768]
        emb = outputs.mean(dim=1)

    return emo_proj(emb)

def extract_long_audio_features(audio):
    chunks = chunk_audio(audio, SR, CHUNK_SEC)
    feats = []

    for c in chunks:
        feats.append(emotion2vec_chunk_embedding(c))

    return torch.mean(torch.stack(feats), dim=0)

# ======================================================
# TRAIN + EVAL
# ======================================================
def run():
    print("\n===== EMOTION2VEC (HF) TEST =====")

    model = LinearProbe().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # -------- TRAIN --------
    for epoch in range(EPOCHS):
        total_loss = 0.0

        for path, label in TRAIN_FILES:
            audio, _ = librosa.load(path, sr=SR)
            features = extract_long_audio_features(audio)

            logits = model(features)
            target = torch.tensor([label]).to(DEVICE)

            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    # -------- TEST --------
    correct = 0
    for path, label in TEST_FILES:
        audio, _ = librosa.load(path, sr=SR)
        features = extract_long_audio_features(audio)

        with torch.no_grad():
            pred = model(features).argmax(dim=1).item()

        correct += int(pred == label)

    print(f"[TEST] EMOTION2VEC-HF → {correct}/{len(TEST_FILES)}")

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    run()
