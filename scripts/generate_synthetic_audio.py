"""Generate synthetic audio data for testing."""
import numpy as np
import soundfile as sf
from pathlib import Path


def generate_crying_audio(duration: float = 3.0, sr: int = 16000) -> np.ndarray:
    """
    Generate synthetic crying-like audio.

    Args:
        duration: Duration in seconds
        sr: Sample rate

    Returns:
        Synthetic audio array

    TODO: Implement synthetic audio generation
    """
    # Simple placeholder - replace with better synthesis
    t = np.linspace(0, duration, int(sr * duration))
    # Multiple frequencies to simulate crying
    audio = (
        0.5 * np.sin(2 * np.pi * 500 * t) +
        0.3 * np.sin(2 * np.pi * 800 * t) +
        0.2 * np.sin(2 * np.pi * 1200 * t)
    )
    return audio


def generate_normal_audio(duration: float = 3.0, sr: int = 16000) -> np.ndarray:
    """
    Generate synthetic normal (non-distress) audio.

    TODO: Implement normal audio synthesis
    """
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 200 * t)
    return audio


if __name__ == "__main__":
    print("Generating synthetic audio data...")

    output_dir = Path("data/synthetic/audio")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate distress samples
    for i in range(10):
        audio = generate_crying_audio()
        sf.write(output_dir / f"distress_{i:03d}.wav", audio, 16000)

    # Generate normal samples
    for i in range(10):
        audio = generate_normal_audio()
        sf.write(output_dir / f"normal_{i:03d}.wav", audio, 16000)

    print(f"Generated 20 synthetic audio files in {output_dir}")
