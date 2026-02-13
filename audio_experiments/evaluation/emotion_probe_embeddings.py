import torch
import torch.nn as nn
import librosa
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# ================= CONFIG =================
SR = 16000
DEVICE = "cpu"
EPOCHS = 5
LR = 1e-3
CHUNK_SEC = 6
NUM_CLASSES = 3

# ================= DATA ===================
TRAIN_FILES = [
    ("audio_experiments/input/calm/calm_long.wav", 0),
    ("audio_experiments/input/fear/fear_long.wav", 1),
    ("audio_experiments/input/cry/cry_long.wav", 2),
]

TEST_FILES = [
    ("audio_experiments/input/calm/calm_medium.wav", 0),
    ("audio_experiments/input/fear/fear_medium.wav", 1),
    ("audio_experiments/input/cry/cry_medium.wav", 2),
]

# ================= CHUNKING =================
def chunk_audio(audio, sr, sec):
    step = sr * sec
    return [audio[i:i+step] for i in range(0, len(audio) - step, step)]

# ================= MODELS ==================
print("Loading baseline wav2vec2...")
wav2vec_base = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).to(DEVICE).eval()
for p in wav2vec_base.parameters():
    p.requires_grad = False

print("Loading emotion-aware wav2vec2...")
emo_processor = Wav2Vec2FeatureExtractor.from_pretrained(
    "r-f/wav2vec-english-speech-emotion-recognition"
)
wav2vec_emo = Wav2Vec2Model.from_pretrained(
    "r-f/wav2vec-english-speech-emotion-recognition"
).to(DEVICE).eval()
for p in wav2vec_emo.parameters():
    p.requires_grad = False

print("Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

proj_base = nn.Linear(768, 256)
proj_emo = nn.Linear(1024, 256)
proj_yam = nn.Linear(1024, 256)

# ================= CLASSIFIER ==============
class EmotionProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

# ================= EMBEDDINGS ===============
def baseline_embed(chunk):
    x = torch.tensor(chunk).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        w = wav2vec_base(x).last_hidden_state.mean(dim=1)
    w = proj_base(w)

    audio_tf = tf.convert_to_tensor(chunk, tf.float32)
    _, y, _ = yamnet(audio_tf)
    y = proj_yam(torch.tensor(y.numpy().mean(axis=0)).unsqueeze(0))

    return torch.cat([w, y], dim=-1)

def emotion_embed(chunk):
    inputs = emo_processor(chunk, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        e = wav2vec_emo(inputs).last_hidden_state.mean(dim=1)
    e = proj_emo(e)

    audio_tf = tf.convert_to_tensor(chunk, tf.float32)
    _, y, _ = yamnet(audio_tf)
    y = proj_yam(torch.tensor(y.numpy().mean(axis=0)).unsqueeze(0))

    return torch.cat([e, y], dim=-1)

def extract_long(audio, mode):
    chunks = chunk_audio(audio, SR, CHUNK_SEC)
    feats = []
    for c in chunks:
        feats.append(baseline_embed(c) if mode == "baseline" else emotion_embed(c))
    return torch.mean(torch.stack(feats), dim=0)

# ================= TRAIN / TEST =============
def run(mode):
    print(f"\n===== {mode.upper()} =====")
    model = EmotionProbe().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(EPOCHS):
        total = 0
        for path, label in TRAIN_FILES:
            audio, _ = librosa.load(path, sr=SR)
            x = extract_long(audio, mode)
            y = torch.tensor([label]).to(DEVICE)

            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {ep+1} | Loss: {total:.3f}")

    correct = 0
    for path, label in TEST_FILES:
        audio, _ = librosa.load(path, sr=SR)
        x = extract_long(audio, mode)
        pred = model(x).argmax(dim=1).item()

        EMO = {0: "CALM", 1: "FEAR", 2: "CRY"}

        print(
            f"[PRED] file={path.split('/')[-1]} | "
            f"true={EMO[label]} | pred={EMO[pred]}"
        )

        correct += int(pred == label)


    print(f"[TEST ACC] {correct}/{len(TEST_FILES)}")

# ================= RUN ======================
if __name__ == "__main__":
    run("baseline")
    run("emotion")
