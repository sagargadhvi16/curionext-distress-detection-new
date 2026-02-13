"""
Script to prune the audio encoder and report parameter reduction.
"""

import torch
from src.audio.encoder import AudioEncoder
from src.audio.pruning import apply_magnitude_pruning, sanity_check

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("\n[INFO] Initializing audio encoder")
    model = AudioEncoder(output_dim=256).to(DEVICE)

    # Dummy audio features (post-feature extraction)
    # Shape aligns with encoder expectations
    dummy_audio = torch.randn(8, 1, 64, 100).to(DEVICE)

    print("\n[INFO] Applying 50% magnitude-based pruning")
    stats = apply_magnitude_pruning(
        model,
        amount=0.5,
        verbose=True
    )

    print("\n[INFO] Running sanity check")
    sanity_check(model, dummy_audio)

    print("\n[OK] Audio encoder pruning completed successfully")
    print("[SUMMARY]", stats)


if __name__ == "__main__":
    main()
