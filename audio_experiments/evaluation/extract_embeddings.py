# ====================== PATH SETUP ======================
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "emotion2vec"))
# =======================================================

import numpy as np
import librosa
import torch

# ====================== CONFIG ======================
SR = 16000
USE_EMOTION2VEC = True    # 🔁 False = baseline | True = +emotion2vec
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===================================================

BASE_DIR = Path(__file__).resolve().parents[1]
CHUNK_DIR = BASE_DIR / "chunks"
OUT_DIR = BASE_DIR / "embeddings"
OUT_DIR.mkdir(exist_ok=True)

# ====================== wav2vec2 ======================
from transformers import Wav2Vec2Processor, Wav2Vec2Model

print("Loading wav2vec2...")
wav2vec_processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)
wav2vec_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
).to(DEVICE).eval()

def extract_wav2vec2(audio):
    inputs = wav2vec_processor(
        audio,
        sampling_rate=SR,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = wav2vec_model(
            inputs.input_values.to(DEVICE)
        )
    emb = outputs.last_hidden_state.mean(dim=1)
    return emb.squeeze(0).cpu().numpy()

# ====================== YAMNet ======================
import tensorflow as tf
import tensorflow_hub as hub

print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def extract_yamnet(audio):
    scores, embeddings, spectrogram = yamnet_model(audio)
    return tf.reduce_mean(embeddings, axis=0).numpy()

# ====================== emotion2vec ======================
if USE_EMOTION2VEC:
    print("Loading emotion2vec...")
    from iemocap_downstream.model import Emotion2Vec
    emotion2vec_model = Emotion2Vec().to(DEVICE).eval()

def extract_emotion2vec(audio):
    x = torch.tensor(audio).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = emotion2vec_model(x)
    return emb.squeeze(0).cpu().numpy()

# ====================== MAIN LOOP ======================
paths = []
embeddings = []

print("Starting embedding extraction...")

for wav_path in CHUNK_DIR.rglob("*.wav"):
    audio, _ = librosa.load(wav_path, sr=SR)

    # skip ultra-short chunks
    if len(audio) < SR:
        continue

    wav2vec_emb = extract_wav2vec2(audio)
    yamnet_emb = extract_yamnet(audio)

    if USE_EMOTION2VEC:
        emotion_emb = extract_emotion2vec(audio)
        final_emb = np.concatenate(
            [wav2vec_emb, yamnet_emb, emotion_emb]
        )
    else:
        final_emb = np.concatenate(
            [wav2vec_emb, yamnet_emb]
        )

    paths.append(str(wav_path))
    embeddings.append(final_emb)

print(f"Total chunks processed: {len(embeddings)}")

# ====================== SAVE ======================
tag = "with_emotion2vec" if USE_EMOTION2VEC else "baseline"
out_path = OUT_DIR / f"embeddings_{tag}.npz"

np.savez(
    out_path,
    paths=np.array(paths),
    embeddings=np.array(embeddings, dtype=object)
)

print(f"Saved embeddings to {out_path}")
