import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Model
import tensorflow_hub as hub
import tensorflow as tf

# ----------------------------
# Config
# ----------------------------
AUDIO_PATHS = [
    "data/raw/audio/train_cry/cry_sample_1.wav",
    "data/synthetic/audio/distress/distress_00034.wav",
    "data/raw/audio/screaming/Screaming/_1JL0dL_h68_out.wav",
    "data/raw/audio/besd/ENGLISH/HAPPY/1.EF_7 happy_1.wav",
    "data/raw/audio/esc50/1-137-A-32.wav",
]

SR = 16000
DEVICE = "cpu"

# ----------------------------
# Models
# ----------------------------
wav2vec2 = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).to(DEVICE).eval()

wav_proj = torch.nn.Linear(768, 256)

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
yam_proj = torch.nn.Linear(1024, 256)

# ----------------------------
# Loop over audios
# ----------------------------
for path in AUDIO_PATHS:
    print(f"\nProcessing: {path}")

    # ---- Load audio ----
    audio, sr = librosa.load(path, sr=SR)
    audio_torch = torch.tensor(audio).unsqueeze(0).to(DEVICE)

    # ---- Wav2Vec2 branch ----
    with torch.no_grad():
        wav_out = wav2vec2(audio_torch).last_hidden_state  # (1, T_w, 768)
        wav_pooled = wav_out.mean(dim=1)                   # (1, 768)
        wav_embed = wav_proj(wav_pooled)                   # (1, 256)

    # ---- YAMNet branch ----
    audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet(audio_tf)
    yam_embed_np = tf.reduce_mean(embeddings, axis=0).numpy()  # (1024,)
    yam_embed = yam_proj(torch.tensor(yam_embed_np).unsqueeze(0))  # (1, 256)

    # ---- Late fusion ----
    fused = torch.cat([wav_embed, yam_embed], dim=-1)  # (1, 512)

    # ---- Print shapes ----
    print("  wav2vec2:", wav_embed.shape)
    print("  yamnet:  ", yam_embed.shape)
    print("  fused:   ", fused.shape)
