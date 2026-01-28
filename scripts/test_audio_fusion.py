from src.fusion.audio_inference import run_test
from src.fusion.audio_model import EmotionClassifier
import torch

TEST_DIR = "data/CurioNext_Audio/test"

model = EmotionClassifier()
model.load_state_dict(torch.load("checkpoints/fusion.pt"))

if __name__ == "__main__":
    run_test(TEST_DIR, model)
