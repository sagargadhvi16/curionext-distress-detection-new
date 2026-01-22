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

LABELS = {
    "cry": 0,
    "fear": 1,
    "anger": 2,
    "neutral": 3,
}

# =====================
# DATA (COLAB PATHS)
# =====================
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
    chunk_len = sr * chunk_sec
    if len(audio) <= chunk_len:
        return [audio]
    return [
        audio[i:i + chunk_len]
        for i in range(0, len(audio) - chunk_len, chunk_len)
    ]

# =====================
# LOAD MODELS
# =====================
print("Loading emotion2vec (FunASR)...")
emo_model = AutoModel(
    model="emotion2vec_base",
    model_revision="v1.0.0",
    device=DEVICE,
)

print("Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

emo_proj = nn.Linear(768, 256).to(DEVICE)
yam_proj = nn.Linear(1024, 256).to(DEVICE)

# =====================
# FEATURE EXTRACTION
# =====================
def emotion2vec_embedding(audio):
    """
    Returns [1, 768]
    """
    out = emo_model.generate(audio, sr=SR)
    emb = np.mean(out[0]["feats"], axis=0)   # (768,)
    emb = torch.from_numpy(emb).float().unsqueeze(0)
    return emb

def yamnet_embedding(audio):
    audio = audio.astype(np.float32)
    audio_tf = tf.convert_to_tensor(audio)
    _, emb, _ = yamnet(audio_tf)
    emb = emb.numpy().mean(axis=0)            # (1024,)
    emb = torch.from_numpy(emb).float().unsqueeze(0)
    return emb

def extract_fused_embedding(audio):
    chunks = chunk_audio(audio, SR, CHUNK_SEC)
    fused = []

    for c in chunks:
        emo = emo_proj(emotion2vec_embedding(c).to(DEVICE))
        yam = yam_proj(yamnet_embedding(c).to(DEVICE))
        fused.append(torch.cat([emo, yam], dim=-1))

    return torch.mean(torch.stack(fused), dim=0)

# =====================
# CLASSIFIER
# =====================
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# =====================
# TRAIN + TEST
# =====================
def run():
    model = EmotionClassifier(len(LABELS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print("\n===== TRAIN =====")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for path, label in TRAIN_FILES:
            audio, _ = librosa.load(path, sr=SR)
            feat = extract_fused_embedding(audio)

            logits = model(feat)
            target = torch.tensor([label]).to(DEVICE)

            loss = loss_fn(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.3f}")

    print("\n===== TEST =====")
    correct = 0
    for path, label in TEST_FILES:
        audio, _ = librosa.load(path, sr=SR)
        feat = extract_fused_embedding(audio)

        with torch.no_grad():
            pred = model(feat).argmax(dim=1).item()

        print(f"{path} → pred: {pred}, gt: {label}")
        correct += int(pred == label)

    print(f"\nAccuracy: {correct}/{len(TEST_FILES)}")

# =====================
if __name__ == "__main__":
    run()
