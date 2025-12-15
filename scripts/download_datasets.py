"""Script to download public datasets for training."""
import os
from pathlib import Path


def download_audio_datasets():
    """
    Download public audio datasets for distress detection.

    Suggested datasets:
    - FSD50K (Freesound Dataset)
    - UrbanSound8K
    - ESC-50 (Environmental Sound Classification)

    TODO: Implement dataset downloading
    """
    print("Downloading audio datasets...")
    # TODO: Add download logic
    pass


def download_biometric_datasets():
    """
    Download public biometric datasets.

    Suggested datasets:
    - PhysioNet datasets
    - WESAD (Wearable Stress and Affect Detection)

    TODO: Implement dataset downloading
    """
    print("Downloading biometric datasets...")
    # TODO: Add download logic
    pass


if __name__ == "__main__":
    print("=" * 50)
    print("CurioNext Dataset Downloader")
    print("=" * 50)

    # Create data directories
    Path("data/raw/audio").mkdir(parents=True, exist_ok=True)
    Path("data/raw/biometric").mkdir(parents=True, exist_ok=True)

    download_audio_datasets()
    download_biometric_datasets()

    print("\nDataset download complete!")
