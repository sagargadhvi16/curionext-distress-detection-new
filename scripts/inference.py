"""
Inference script for audio-only distress detection.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np

from src.audio.encoder import AudioEncoder
from src.audio.classifier import AudioDistressClassifier
from src.audio.preprocessing import load_audio, normalize_audio
from src.audio.features import extract_mfcc
THRESHOLD = 0.80   # start strict


# -------------------------------------------------
# Argument parsing
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Audio Distress Inference")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input audio file (.wav)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/audio_distress_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load models
    encoder = AudioEncoder().to(device)
    classifier = AudioDistressClassifier().to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    classifier.load_state_dict(checkpoint["classifier"])

    encoder.eval()
    classifier.eval()

    # Load & preprocess audio
    audio, sr = load_audio(args.audio)
    audio = normalize_audio(audio)

    mfcc = extract_mfcc(audio, sr)      # (39, T)
    mfcc = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0)
    # Shape: (1, 1, 39, T)

    mfcc = mfcc.to(device)

    # Inference
    with torch.no_grad():
        embedding = encoder(mfcc)
        logit = classifier(embedding)
        prob = torch.sigmoid(logit).item()

    prediction = "DISTRESS" if prob >= THRESHOLD else "NON-DISTRESS"

    print("\n==============================")
    print(f"Distress probability: {prob:.4f}")
    print(f"Threshold used     : {THRESHOLD}")
    print(f"Prediction         : {prediction}")
    print("==============================\n")



if __name__ == "__main__":
    main()

