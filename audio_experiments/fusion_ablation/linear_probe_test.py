import torch
import torch.nn as nn
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from transformers import Wav2Vec2Model

# ======================================================
# CONFIG (SMALL CONFIRMATION SET)
# ======================================================
AUDIO_FILES = [
    ("data/raw/audio/train_cry/cry_sample_1.wav", 1),
    ("data/synthetic/audio/distress/distress_00034.wav", 1),
    ("data/raw/audio/screaming/Screaming/_1JL0dL_h68_out.wav", 1),
    ("data/raw/audio/besd/ENGLISH/HAPPY/1.EF_7 happy_1.wav", 0),
    ("data/raw/audio/esc50/1-137-A-32.wav", 0),
]

SR = 16000
DEVICE = "cpu"
EPOCHS = 5
LR = 1e-3

# ======================================================
# MODELS (FROZEN)
# ======================================================

# ---- wav2vec2 ----
wav2vec2 = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).to(DEVICE).eval()

for p in wav2vec2.parameters():
    p.requires_grad = False

wav_proj = nn.Linear(768, 256)

# ---- YAMNet ----
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
yam_proj = nn.Linear(1024, 256)

# ======================================================
# LINEAR PROBE CLASSIFIER
# ======================================================
class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_features(audio_np):
    audio_torch = torch.tensor(audio_np).unsqueeze(0).to(DEVICE)

    # wav2vec2
    with torch.no_grad():
        wav_out = wav2vec2(audio_torch).last_hidden_state
        wav_embed = wav_proj(wav_out.mean(dim=1))  # (1, 256)

    # YAMNet
    audio_tf = tf.convert_to_tensor(audio_np, dtype=tf.float32)
    _, yam_embeddings, _ = yamnet(audio_tf)
    yam_embed = yam_proj(
        torch.tensor(yam_embeddings.numpy().mean(axis=0)).unsqueeze(0)
    )

    return wav_embed, yam_embed

# ======================================================
# TRAIN FUNCTION
# ======================================================
def train_probe(mode):
    print(f"\n===== TRAINING MODE: {mode.upper()} =====")

    input_dim = 256 if mode != "fusion" else 512
    probe = LinearProbe(input_dim).to(DEVICE)

    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for path, label in AUDIO_FILES:
            audio, _ = librosa.load(path, sr=SR)

            wav_embed, yam_embed = extract_features(audio)

            if mode == "wav2vec2":
                features = wav_embed
            elif mode == "yamnet":
                features = yam_embed
            else:
                features = torch.cat([wav_embed, yam_embed], dim=-1)

            logits = probe(features)
            target = torch.tensor([label]).to(DEVICE)

            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss:.4f}")

    return probe

# ======================================================
# EVALUATION FUNCTION
# ======================================================
def evaluate(probe, mode):
    probe.eval()
    correct = 0

    for path, label in AUDIO_FILES:
        audio, _ = librosa.load(path, sr=SR)

        wav_embed, yam_embed = extract_features(audio)

        if mode == "wav2vec2":
            features = wav_embed
        elif mode == "yamnet":
            features = yam_embed
        else:
            features = torch.cat([wav_embed, yam_embed], dim=-1)

        with torch.no_grad():
            pred = probe(features).argmax(dim=1).item()

        if pred == label:
            correct += 1

    print(f"[EVAL] {mode.upper()} : {correct}/{len(AUDIO_FILES)} correct")

# ======================================================
# RUN ABLATION
# ======================================================
if __name__ == "__main__":
    for mode in ["wav2vec2", "yamnet", "fusion"]:
        probe = train_probe(mode)
        evaluate(probe, mode)
