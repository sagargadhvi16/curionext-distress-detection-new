"""Audio preprocessing utilities."""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple


def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 16000 Hz for YAMNet)

    Returns:
        Tuple of (audio_data, sample_rate)

    TODO: Implement audio loading with error handling
    """
    path = Path(file_path) #convert string path to path object
    
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        # Fast validation to catch corrupted files
        sf.info(str(path))

        audio, sample_rate = librosa.load(
            str(path),
            sr=sr, #resampling happens here,resamples internally to sr=16000
            mono=True
        )

        if audio.size == 0:
            raise ValueError("Loaded audio is empty")

        return audio, sample_rate

    except RuntimeError:
        raise ValueError(f"Corrupted or unsupported audio file: {file_path}")

    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}")

def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000
) -> np.ndarray:
    """
    Resample audio to target sample rate using high-quality resampling.

    Args:
        audio: Audio waveform array
        orig_sr: Original sampling rate
        target_sr: Target sampling rate (default: 16000 Hz)

    Returns:
        Resampled audio array
    """
    if audio.size == 0:
        raise ValueError("Cannot resample empty audio")

    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive")

    # No resampling needed
    if orig_sr == target_sr:
        return audio

    try:
        resampled_audio = librosa.resample(
            audio,
            orig_sr=orig_sr,
            target_sr=target_sr
        )
        return resampled_audio

    except Exception as e:
        raise ValueError(f"Failed to resample audio: {e}")

def normalize_audio(audio: np.ndarray, method: str = "peak") -> np.ndarray:
    """
    Normalize audio using peak or RMS normalization.

    Args:
        audio: Audio signal array
        method: Normalization method ("peak" or "rms")

    Returns:
        Normalized audio array

    TODO: Implement normalization methods
    """
    if audio.size == 0:
        raise ValueError("Cannot normalize empty audio")

    if method == "peak":
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        return audio / peak

    elif method == "rms":
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return audio
        return audio / rms

    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def detect_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: int = 20
):
    """
    Detect silent regions in audio based on energy threshold.

    Args:
        audio: Audio waveform array
        sr: Sample rate
        threshold_db: Silence threshold in dB below peak

    Returns:
        List of (start_sample, end_sample) tuples representing silence regions
    """
    if audio.size == 0:
        raise ValueError("Cannot detect silence in empty audio")

    try:
        # Non-silent regions
        non_silent_intervals = librosa.effects.split(
            audio,
            top_db=threshold_db
        )

        silence_intervals = []
        prev_end = 0

        for start, end in non_silent_intervals:
            if start > prev_end:
                silence_intervals.append((prev_end, start))
            prev_end = end

        # Trailing silence
        if prev_end < len(audio):
            silence_intervals.append((prev_end, len(audio)))

        return silence_intervals

    except Exception as e:
        raise ValueError(f"Failed to detect silence: {e}")


def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: int = 20
) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.

    Args:
        audio: Audio signal array
        sr: Sample rate
        top_db: Threshold in dB below reference to consider as silence

    Returns:
        Trimmed audio array

    TODO: Implement silence trimming
    """
    if audio.size == 0:
        raise ValueError("Cannot trim silence from empty audio")

    try:
        trimmed_audio, _ = librosa.effects.trim(
            audio,
            top_db=top_db
        )
        return trimmed_audio

    except Exception as e:
        raise ValueError(f"Failed to trim silence: {e}")
