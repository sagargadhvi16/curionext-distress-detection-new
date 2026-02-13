"""
Minimal compatibility test:
Wav2Vec2 + YAMNet with enforced 16kHz resampling and concatenation.
"""

import torch
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

AUDIO_FILE = "data/raw/audio/train_cry/cry_sample_1.wav"
TARGET_SR = 16000


def main():
    print("🔹 Loading audio")
    audio, sr = librosa.load(AUDIO_FILE, sr=None)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # -------------------------------------------------
    # Wav2Vec2 (PyTorch)
    # -------------------------------------------------
    print("🔹 Loading Wav2Vec2")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        w2v_out = w2v_model(**inputs).last_hidden_state.mean(dim=1)
        # shape: (1, 768)

    print("Wav2Vec2 embedding:", w2v_out.shape)

    # -------------------------------------------------
    # YAMNet (TensorFlow)
    # -------------------------------------------------
    print("🔹 Loading YAMNet")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet(waveform)
    yamnet_emb = tf.reduce_mean(embeddings, axis=0)
    yamnet_emb = torch.from_numpy(yamnet_emb.numpy()).unsqueeze(0)
    # shape: (1, 1024)

    print("YAMNet embedding:", yamnet_emb.shape)

    # -------------------------------------------------
    # Concatenation
    # -------------------------------------------------
    fused = torch.cat([w2v_out, yamnet_emb], dim=1)
    print("✅ Fused embedding shape:", fused.shape)
    print("🎯 SUCCESS: Concatenation works")


if __name__ == "__main__":
    main()
