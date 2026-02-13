from src.fusion.audio_trainer import train
import torch
from pathlib import Path


TRAIN_DIR = "data/CurioNext_Audio/train"
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    model = train(
        train_dir=TRAIN_DIR,
        epochs=15,
        device="cpu"
    )

    torch.save(model.state_dict(), CKPT_DIR / "fusion.pt")
    print("Model saved to checkpoints/fusion.pt")
