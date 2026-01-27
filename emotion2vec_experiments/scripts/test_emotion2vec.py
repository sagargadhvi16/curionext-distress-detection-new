import torch
import librosa
from src.encoders.emotion2vec_encoder import Emotion2VecEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

encoder = Emotion2VecEncoder(device=DEVICE)

audio, _ = librosa.load("data/raw_audio/sample.wav", sr=16000)
audio = audio[:16000 * 2]  # 2 seconds

x = torch.tensor(audio).unsqueeze(0).to(DEVICE)

emb = encoder.encode(x)

print("Embedding shape:", emb.shape)
print("Embedding dtype:", emb.dtype)
