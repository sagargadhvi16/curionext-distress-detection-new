import os
import torch
import librosa
import numpy as np
from src.encoders.emotion2vec_encoder import Emotion2VecEncoder

AUDIO_DIR = "data/raw_audio"
OUT_DIR = "data/embeddings/emotion2vec"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
encoder = Emotion2VecEncoder(device=DEVICE)

for fname in os.listdir(AUDIO_DIR):
    if not fname.endswith(".wav"):
        continue

    audio, _ = librosa.load(os.path.join(AUDIO_DIR, fname), sr=16000)
    x = torch.tensor(audio).unsqueeze(0).to(DEVICE)

    emb = encoder.encode(x).cpu().numpy()
    np.save(os.path.join(OUT_DIR, fname.replace(".wav", ".npy")), emb)

    print(f"Saved: {fname}")
