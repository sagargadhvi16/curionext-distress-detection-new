import torch
import torch.nn as nn
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from transformers import Wav2Vec2Model

# ======================================================
# DATA SPLIT
# ======================================================
TRAIN_FILES = [
    ("data/raw/audio/train_cry/cry_sample_1.wav", 1),
    ("data/raw/audio/train_cry/cry_sample_6.wav", 1),
    ("data/synthetic/audio/distress/distress_00034.wav", 1),
    ("data/synthetic/audio/distress/distress_00006.wav", 1),
    ("data/raw/audio/screaming/Screaming/_1JL0dL_h68_out.wav", 1),
    ("data/raw/audio/screaming/Screaming/_LyX0W1PAgw_out.wav", 1),
    ("data/raw/audio/screaming/NotScreaming/_0bN5mYLXb0_out.wav", 0),
    ("data/raw/audio/screaming/NotScreaming/_3zEO-HtX8o_out.wav", 0),
    ("data/raw/audio/besd/ENGLISH/ANGER/1.EF_12 Angry_1.wav", 1),
    ("data/raw/audio/besd/ENGLISH/HAPPY/1.EF_7 happy_1.wav", 0),
    ("data/raw/audio/besd/ENGLISH/HAPPY/2.EF_8 happy_4.wav", 0),
    ("data/raw/audio/besd/ENGLISH/HAPPY/5.EF_8 happy_5.wav", 0),
    ("data/raw/audio/esc50/1-137-A-32.wav", 0),
    ("data/raw/audio/esc50/1-4211-A-12.wav", 0),
    ("data/raw/audio/esc50/1-9841-A-13.wav", 0),
    ("data/raw/audio/esc50/1-16568-A-3.wav", 0),
    ("data/raw/audio/esc50/1-18655-A-31.wav", 0),

]

TEST_FILES = [
    ("data/raw/audio/esc50/1-977-A-39.wav", 0),
]

SR = 16000
DEVICE = "cpu"
EPOCHS = 5
LR = 1e-3

# ======================================================
# MODELS (FROZEN)
# ======================================================
wav2vec2 = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).to(DEVICE).eval()

for p in wav2vec2.parameters():
    p.requires_grad = False

wav_proj = nn.Linear(768, 256)

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
yam_proj = nn.Linear(1024, 256)

# ======================================================
# CLASSIFIER
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
def extract_features(audio_np, use_raw=False):
    audio_torch = torch.tensor(audio_np).unsqueeze(0).to(DEVICE)

    # wav2vec2
    with torch.no_grad():
        wav_out = wav2vec2(audio_torch).last_hidden_state.mean(dim=1)  # (1, 768)

    wav_embed_768 = wav_out
    wav_embed_256 = wav_proj(wav_out)

    # YAMNet
    audio_tf = tf.convert_to_tensor(audio_np, dtype=tf.float32)
    _, yam_embeddings, _ = yamnet(audio_tf)
    yam_raw = torch.tensor(yam_embeddings.numpy().mean(axis=0)).unsqueeze(0)

    yam_embed_1024 = yam_raw
    yam_embed_256 = yam_proj(yam_raw)

    return wav_embed_768, wav_embed_256, yam_embed_1024, yam_embed_256

# ======================================================
# TRAIN + EVAL
# ======================================================
def run_experiment(mode):
    print(f"\n===== MODE: {mode.upper()} =====")

    if mode == "wav2vec2_768":
        input_dim = 768
    elif mode == "yamnet_1024":
        input_dim = 1024
    elif mode in ["wav2vec2_256", "yamnet_256"]:
        input_dim = 256
    else:  # fusion_512
        input_dim = 512

    probe = LinearProbe(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # -------- TRAIN --------
    for epoch in range(EPOCHS):
        total_loss = 0.0

        for path, label in TRAIN_FILES:
            audio, _ = librosa.load(path, sr=SR)
            w768, w256, y1024, y256 = extract_features(audio)

            if mode == "wav2vec2_768":
                features = w768
            elif mode == "wav2vec2_256":
                features = w256
            elif mode == "yamnet_1024":
                features = y1024
            elif mode == "yamnet_256":
                features = y256
            else:  # fusion
                features = torch.cat([w256, y256], dim=-1)

            logits = probe(features)
            target = torch.tensor([label]).to(DEVICE)

            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # -------- EVAL --------
    correct = 0
    for path, label in TEST_FILES:
        audio, _ = librosa.load(path, sr=SR)
        w768, w256, y1024, y256 = extract_features(audio)

        if mode == "wav2vec2_768":
            features = w768
        elif mode == "wav2vec2_256":
            features = w256
        elif mode == "yamnet_1024":
            features = y1024
        elif mode == "yamnet_256":
            features = y256
        else:
            features = torch.cat([w256, y256], dim=-1)

        with torch.no_grad():
            pred = probe(features).argmax(dim=1).item()

        if pred == label:
            correct += 1

    print(f"[HELD-OUT TEST] {mode.upper()} : {correct}/{len(TEST_FILES)}")

# ======================================================
# RUN ALL COMPARISONS
# ======================================================
if __name__ == "__main__":
    MODES = [
        "wav2vec2_768",
        "wav2vec2_256",
        "yamnet_1024",
        "yamnet_256",
        "fusion_512",
    ]

    for m in MODES:
        run_experiment(m)
