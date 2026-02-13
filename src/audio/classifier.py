import torch
import torch.nn as nn

class AudioDistressClassifier(nn.Module):
    #binary classifier on top of audio embeddings
    def __init__(self, input_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x:(B,256)
        return self.net(x).squeeze(-1)  # (B,)
