"""
Compatibility test for Wav2Vec2 (PyTorch) + YAMNet (TensorFlow).
Windows-safe version.
"""

import os
import numpy as np
import torch
import librosa
import tensorflow as tf
import tensorflow_hub as hub

from transformers import Wav2Vec2Processor, Wav2Vec2Model

SR = 16000

TEST_AUDIO_FILES = [
    "data/raw/audio/train_cry/cry_sample_1.wav",
    "data/synthetic/audio/distress/distress_00034.wav",
    "data/raw/audio/screaming/Screaming/_1JL0dL_h68_out.wav",
    "data/raw/audio/besd/ENGLISH/HAPPY/1.EF_7 happy_1.wav",
    "data/raw/audio/esc50/1-137-A-32.wav",
]

def load_audio(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return audio.astype(np.float32)

def main():
    print("\n🔹 Loading Wav2Vec2 (PyTorch)")
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    wav2vec = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    wav2vec.eval()

    print("🔹 Loading YAMNet (TensorFlow)")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    print("\n=========== COMPATIBILITY CHECK ===========\n")

    for path in TEST_AUDIO_FILES:
        print(f"▶ {path}")

        if not os.path.exists(path):
            print("  ❌ File not found\n")
            continue

        audio = load_audio(path)

        # ---- Wav2Vec2 ----
        with torch.no_grad():
            inputs = processor(
                audio,
                sampling_rate=SR,
                return_tensors="pt",
                padding=True
            )
            w2v_out = wav2vec(**inputs)
            w2v_emb = w2v_out.last_hidden_state  # (1, T, 768)

        # ---- YAMNet ----
        audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, yam_emb, spectrogram = yamnet(audio_tf)
        # yam_emb: (frames, 1024)

        print(f"  Wav2Vec2 → {tuple(w2v_emb.shape)}")
        print(f"  YAMNet   → {tuple(yam_emb.shape)}")
        print("  ✅ OK\n")

    print("🎯 SUCCESS: Wav2Vec2 + YAMNet are compatible")
    print("===========================================\n")

if __name__ == "__main__":
    main()
