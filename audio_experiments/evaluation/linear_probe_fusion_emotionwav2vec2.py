import torch
import torch.nn as nn
import librosa
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# ======================================================
# CONFIG
# ======================================================
SR = 16000
DEVICE = "cpu"
EPOCHS = 5
LR = 1e-3
CHUNK_SEC = 6   # chunk size for long audio

# ======================================================
# LONG-AUDIO DATA
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
    chunk_len = chunk_sec * sr
    chunks = []
    for i in range(0, len(audio) - chunk_len, chunk_len):
        chunks.append(audio[i:i + chunk_len])
    return chunks

# ======================================================
# LOAD MODELS (EXPLICIT DIFFERENCE)
# ======================================================
print("Loading BASELINE wav2vec2...")
wav2vec_base = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).to(DEVICE).eval()

for p in wav2vec_base.parameters():
    p.requires_grad = False

print("Loading EMOTION-AWARE wav2vec2...")
emo_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "r-f/wav2vec-english-speech-emotion-recognition"
)

wav2vec_emo = Wav2Vec2Model.from_pretrained(
    "r-f/wav2vec-english-speech-emotion-recognition"
).to(DEVICE).eval()

for p in wav2vec_emo.parameters():
    p.requires_grad = False

# Projection layers (same dimensionality for fairness)
proj_base = nn.Linear(768, 256)
proj_emo = nn.Linear(1024, 256)

print("Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
yam_proj = nn.Linear(1024, 256)

# ======================================================
# LINEAR PROBE (UNCHANGED)
# ======================================================
class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# ======================================================
# FEATURE EXTRACTION (PER CHUNK)
# ======================================================
def baseline_chunk_embedding(chunk):
    audio_t = torch.tensor(chunk).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        w = wav2vec_base(audio_t).last_hidden_state.mean(dim=1)
    w = proj_base(w)

    audio_tf = tf.convert_to_tensor(chunk, dtype=tf.float32)
    _, y_emb, _ = yamnet(audio_tf)
    y = yam_proj(torch.tensor(y_emb.numpy().mean(axis=0)).unsqueeze(0))

    return torch.cat([w, y], dim=-1)


def emotion_chunk_embedding(chunk):
    emo_inputs = emo_feature_extractor(
        chunk, sampling_rate=SR, return_tensors="pt"
    ).input_values.to(DEVICE)

    with torch.no_grad():
        e = wav2vec_emo(emo_inputs).last_hidden_state.mean(dim=1)
    e = proj_emo(e)

    audio_tf = tf.convert_to_tensor(chunk, dtype=tf.float32)
    _, y_emb, _ = yamnet(audio_tf)
    y = yam_proj(torch.tensor(y_emb.numpy().mean(axis=0)).unsqueeze(0))

    return torch.cat([e, y], dim=-1)

# ======================================================
# AGGREGATE LONG AUDIO
# ======================================================
def extract_long_audio_features(audio, mode):
    chunks = chunk_audio(audio, SR, CHUNK_SEC)
    feats = []

    for c in chunks:
        if mode == "baseline":
            feats.append(baseline_chunk_embedding(c))
        else:
            feats.append(emotion_chunk_embedding(c))

    return torch.mean(torch.stack(feats), dim=0)

# ======================================================
# TRAIN + EVAL
# ======================================================
def run_experiment(mode):
    print(f"\n===== RUNNING: {mode.upper()} =====")

    model = LinearProbe().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # -------- TRAIN --------
    for epoch in range(EPOCHS):
        total_loss = 0.0

        for path, label in TRAIN_FILES:
            audio, _ = librosa.load(path, sr=SR)
            features = extract_long_audio_features(audio, mode)

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
        features = extract_long_audio_features(audio, mode)

        with torch.no_grad():
            pred = model(features).argmax(dim=1).item()

        if pred == label:
            correct += 1

    print(f"[TEST] {mode.upper()} → {correct}/{len(TEST_FILES)}")

# ======================================================
# MAIN — TWO EXPLICIT EXPERIMENTS
# ======================================================
if __name__ == "__main__":
    run_experiment("baseline")
    run_experiment("emotion")
