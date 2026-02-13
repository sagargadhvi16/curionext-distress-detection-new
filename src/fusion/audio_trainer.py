import os
import torch
import librosa
from torch import nn
from src.fusion.audio_embeddings import FusionEmbeddings
from src.fusion.audio_model import EmotionClassifier


SR = 16000
TRAIN_WINDOW_SEC = 2

LABEL_MAP = {
    "cry": 0,
    "fear": 1,
    "anger": 2,
    "neutral": 3,
    "abuse": 4,
    "scream": 5
}

def chunk_audio(audio, sr, win_sec):
    win_len = sr * win_sec
    return [
        audio[i:i+win_len]
        for i in range(0, len(audio)-win_len+1, win_len)
    ]

def train(train_dir, epochs=15, device="cpu"):
    embedder = FusionEmbeddings().to(device)
    model = EmotionClassifier().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for emo, label in LABEL_MAP.items():
            emo_dir = os.path.join(train_dir, emo)

            for f in os.listdir(emo_dir)[:4]:
                audio, _ = librosa.load(os.path.join(emo_dir, f), sr=SR)
                windows = chunk_audio(audio, SR, TRAIN_WINDOW_SEC)

                for w in windows[:10]:
                    emb = embedder(w).unsqueeze(0).to(device)
                    target = torch.tensor([label], device=device)

                    loss = loss_fn(model(emb), target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    return model
