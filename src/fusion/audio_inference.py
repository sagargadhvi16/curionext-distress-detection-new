import os, json
import torch
import librosa
from src.fusion.audio_embeddings import FusionEmbeddings
from src.fusion.audio_model import EmotionClassifier

SR = 16000
TEST_WINDOW_SEC = 10

ID2LABEL = {
    0: "cry",
    1: "fear",
    2: "anger",
    3: "neutral",
    4: "abuse",
    5: "scream"
}

def chunk_audio(audio, sr, win_sec):
    win_len = sr * win_sec
    return [
        audio[i:i+win_len]
        for i in range(0, len(audio)-win_len+1, win_len)
    ]

def run_test(test_dir, model, device="cpu"):
    embedder = FusionEmbeddings().to(device)
    model.eval()

    for wav_file in sorted(os.listdir(test_dir)):
        if not wav_file.endswith(".wav"):
            continue

        base = wav_file.replace(".wav", "")
        audio, _ = librosa.load(os.path.join(test_dir, wav_file), sr=SR)
        windows = chunk_audio(audio, SR, TEST_WINDOW_SEC)

        preds = []
        with torch.no_grad():
            for w in windows:
                emb = embedder(w).unsqueeze(0).to(device)
                preds.append(ID2LABEL[model(emb).argmax(dim=1).item()])

        print(wav_file, preds)
