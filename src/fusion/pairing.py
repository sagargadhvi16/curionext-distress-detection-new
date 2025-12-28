"""Multimodal sample pairing utilities for audio and biometric data."""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AudioSample:
    """Audio sample metadata."""
    file_path: Path
    duration: Optional[float] = None
    timestamp: Optional[datetime] = None
    label: Optional[str] = None
    sample_rate: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def from_file(cls, file_path: Path, metadata_file: Optional[Path] = None) -> 'AudioSample':
        """
        Create AudioSample from file path.
        
        Args:
            file_path: Path to audio file
            metadata_file: Optional path to metadata JSON file
            
        Returns:
            AudioSample instance
        """
        sample = cls(file_path=file_path)
        
        # Try to load metadata if provided
        if metadata_file and metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    sample.metadata = metadata
                    sample.duration = metadata.get('duration_sec')
                    sample.label = metadata.get('label')
                    sample.sample_rate = metadata.get('sample_rate', 16000)
                    
                    # Parse timestamp if available
                    if 'timestamp' in metadata:
                        timestamp_str = metadata['timestamp']
                        try:
                            sample.timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        except (ValueError, AttributeError):
                            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            except Exception as e:
                logger.warning(f"Error loading metadata from {metadata_file}: {e}")
        
        return sample


@dataclass
class BiometricSample:
    """Biometric sample metadata (HRV + Accelerometer)."""
    hrv_file_path: Optional[Path] = None
    accel_file_path: Optional[Path] = None
    timestamp: Optional[datetime] = None
    label: Optional[str] = None
    duration: Optional[float] = None
    hrv_data: Optional[Dict[str, Any]] = None
    accel_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def from_files(
        cls,
        hrv_file: Optional[Path] = None,
        accel_file: Optional[Path] = None
    ) -> 'BiometricSample':
        """
        Create BiometricSample from file paths.
        
        Args:
            hrv_file: Path to HRV JSON file
            accel_file: Path to accelerometer JSON file
            
        Returns:
            BiometricSample instance
        """
        sample = cls(hrv_file_path=hrv_file, accel_file_path=accel_file)
        
        # Load HRV data
        if hrv_file and hrv_file.exists():
            try:
                with open(hrv_file, 'r', encoding='utf-8') as f:
                    sample.hrv_data = json.load(f)
                    sample.metadata['hrv'] = sample.hrv_data
                    
                    # Extract common fields
                    if 'duration_sec' in sample.hrv_data:
                        sample.duration = sample.hrv_data['duration_sec']
                    if 'label' in sample.hrv_data:
                        sample.label = sample.hrv_data.get('label')
                    if 'timestamp' in sample.hrv_data:
                        try:
                            timestamp_str = sample.hrv_data['timestamp']
                            sample.timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        except (ValueError, AttributeError):
                            logger.warning(f"Could not parse HRV timestamp: {timestamp_str}")
            except Exception as e:
                logger.warning(f"Error loading HRV data from {hrv_file}: {e}")
        
        # Load accelerometer data
        if accel_file and accel_file.exists():
            try:
                with open(accel_file, 'r', encoding='utf-8') as f:
                    sample.accel_data = json.load(f)
                    sample.metadata['accelerometer'] = sample.accel_data
                    
                    # Use accel timestamp if HRV doesn't have one
                    if sample.timestamp is None and 'timestamp' in sample.accel_data:
                        try:
                            timestamp_str = sample.accel_data['timestamp']
                            sample.timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        except (ValueError, AttributeError):
                            pass
                    
                    # Use accel duration if HRV doesn't have one
                    if sample.duration is None and 'duration_sec' in sample.accel_data:
                        sample.duration = sample.accel_data['duration_sec']
                    
                    # Use accel label if HRV doesn't have one
                    if sample.label is None and 'label' in sample.accel_data:
                        sample.label = sample.accel_data.get('label')
            except Exception as e:
                logger.warning(f"Error loading accelerometer data from {accel_file}: {e}")
        
        return sample


@dataclass
class PairedSample:
    """Paired audio and biometric sample."""
    audio: AudioSample
    biometric: BiometricSample
    pairing_method: str  # 'timestamp', 'metadata', 'index', etc.
    pairing_score: float = 1.0  # Confidence score (0.0 to 1.0)
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def pair_by_timestamp(
    audio_samples: List[AudioSample],
    biometric_samples: List[BiometricSample],
    time_tolerance_seconds: float = 5.0
) -> List[Tuple[AudioSample, BiometricSample, float]]:
    """
    Pair samples by timestamp with tolerance.
    
    Args:
        audio_samples: List of audio samples
        biometric_samples: List of biometric samples
        time_tolerance_seconds: Maximum time difference for pairing (seconds)
        
    Returns:
        List of tuples (audio, biometric, score) where score is 1.0 if within tolerance,
        decreasing based on time difference
    """
    pairs = []
    used_audio_indices = set()
    used_bio_indices = set()
    
    # Create list of samples with timestamps
    audio_with_ts = [(i, a) for i, a in enumerate(audio_samples) if a.timestamp is not None]
    bio_with_ts = [(i, b) for i, b in enumerate(biometric_samples) if b.timestamp is not None]
    
    # Sort by timestamp
    audio_with_ts.sort(key=lambda x: x[1].timestamp)
    bio_with_ts.sort(key=lambda x: x[1].timestamp)
    
    # Match by nearest timestamp
    for audio_idx, audio in audio_with_ts:
        if audio_idx in used_audio_indices:
            continue
        
        best_bio_idx = None
        best_time_diff = float('inf')
        
        for bio_idx, bio in bio_with_ts:
            if bio_idx in used_bio_indices:
                continue
            
            time_diff = abs((audio.timestamp - bio.timestamp).total_seconds())
            
            if time_diff < best_time_diff and time_diff <= time_tolerance_seconds:
                best_time_diff = time_diff
                best_bio_idx = bio_idx
        
        if best_bio_idx is not None:
            used_audio_indices.add(audio_idx)
            used_bio_indices.add(best_bio_idx)
            
            # Score based on time difference (closer = higher score)
            score = max(0.0, 1.0 - (best_time_diff / time_tolerance_seconds))
            pairs.append((audio, biometric_samples[best_bio_idx], score))
    
    return pairs


def pair_by_metadata(
    audio_samples: List[AudioSample],
    biometric_samples: List[BiometricSample]
) -> List[Tuple[AudioSample, BiometricSample, float]]:
    """
    Pair samples by metadata matching (label, duration similarity).
    
    Args:
        audio_samples: List of audio samples
        biometric_samples: List of biometric samples
        
    Returns:
        List of tuples (audio, biometric, score) based on metadata similarity
    """
    pairs = []
    used_audio_indices = set()
    used_bio_indices = set()
    
    # First, try to match by label
    audio_by_label = {}
    bio_by_label = {}
    
    for i, audio in enumerate(audio_samples):
        label = audio.label
        if label:
            if label not in audio_by_label:
                audio_by_label[label] = []
            audio_by_label[label].append(i)
    
    for i, bio in enumerate(biometric_samples):
        label = bio.label
        if label:
            if label not in bio_by_label:
                bio_by_label[label] = []
            bio_by_label[label].append(i)
    
    # Match by label first
    for label in set(audio_by_label.keys()) & set(bio_by_label.keys()):
        audio_indices = [idx for idx in audio_by_label[label] if idx not in used_audio_indices]
        bio_indices = [idx for idx in bio_by_label[label] if idx not in used_bio_indices]
        
        # Match in order, prioritizing similar durations if available
        for audio_idx in audio_indices[:len(bio_indices)]:
            if len(bio_indices) == 0:
                break
            
            audio = audio_samples[audio_idx]
            
            # Find best matching biometric by duration similarity
            best_bio_idx = None
            best_duration_diff = float('inf')
            
            for bio_idx in bio_indices:
                bio = biometric_samples[bio_idx]
                
                if audio.duration is not None and bio.duration is not None:
                    duration_diff = abs(audio.duration - bio.duration)
                    if duration_diff < best_duration_diff:
                        best_duration_diff = duration_diff
                        best_bio_idx = bio_idx
                else:
                    # No duration info, just use first available
                    best_bio_idx = bio_idx
                    break
            
            if best_bio_idx is not None:
                used_audio_indices.add(audio_idx)
                used_bio_indices.remove(best_bio_idx)
                bio_indices.remove(best_bio_idx)
                
                # Score based on label match and duration similarity
                score = 0.8  # Base score for label match
                if audio.duration and biometric_samples[best_bio_idx].duration:
                    duration_match = 1.0 - min(1.0, best_duration_diff / max(audio.duration, biometric_samples[best_bio_idx].duration))
                    score = 0.5 + 0.5 * duration_match
                
                pairs.append((audio, biometric_samples[best_bio_idx], score))
    
    return pairs


def pair_by_index(
    audio_samples: List[AudioSample],
    biometric_samples: List[BiometricSample]
) -> List[Tuple[AudioSample, BiometricSample, float]]:
    """
    Pair samples by index (simple sequential pairing).
    
    Args:
        audio_samples: List of audio samples
        biometric_samples: List of biometric samples
        
    Returns:
        List of tuples (audio, biometric, score) paired by index
    """
    pairs = []
    min_length = min(len(audio_samples), len(biometric_samples))
    
    for i in range(min_length):
        pairs.append((audio_samples[i], biometric_samples[i], 0.5))  # Lower score for index pairing
    
    return pairs


def pair_multimodal_samples(
    audio_list: List[AudioSample],
    bio_list: List[BiometricSample],
    pairing_strategy: str = 'auto',
    time_tolerance_seconds: float = 5.0
) -> List[PairedSample]:
    """
    Pair audio and biometric samples by timestamp/metadata.
    
    This is the main function for pairing multimodal samples. It tries multiple
    pairing strategies in order of preference:
    1. Timestamp-based pairing (if timestamps available)
    2. Metadata-based pairing (by label, duration)
    3. Index-based pairing (fallback)
    
    Args:
        audio_list: List of AudioSample objects
        bio_list: List of BiometricSample objects
        pairing_strategy: Pairing strategy ('auto', 'timestamp', 'metadata', 'index')
        time_tolerance_seconds: Maximum time difference for timestamp pairing (seconds)
        
    Returns:
        List of PairedSample objects
        
    Example:
        >>> audio_samples = [AudioSample.from_file(Path("audio1.wav"))]
        >>> bio_samples = [BiometricSample.from_files(hrv_file=Path("hrv1.json"))]
        >>> paired = pair_multimodal_samples(audio_samples, bio_samples)
        >>> print(f"Paired {len(paired)} samples")
    """
    logger.info(f"Pairing {len(audio_list)} audio samples with {len(bio_list)} biometric samples")
    
    if len(audio_list) == 0 or len(bio_list) == 0:
        logger.warning("Empty audio or biometric list, returning empty pairs")
        return []
    
    pairs = []
    used_audio = set()
    used_bio = set()
    
    # Strategy 1: Timestamp-based pairing (if requested or auto)
    if pairing_strategy in ('auto', 'timestamp'):
        timestamp_pairs = pair_by_timestamp(audio_list, bio_list, time_tolerance_seconds)
        for audio, bio, score in timestamp_pairs:
            audio_idx = audio_list.index(audio)
            bio_idx = bio_list.index(bio)
            used_audio.add(audio_idx)
            used_bio.add(bio_idx)
            pairs.append(PairedSample(
                audio=audio,
                biometric=bio,
                pairing_method='timestamp',
                pairing_score=score
            ))
        
        if pairs and pairing_strategy == 'timestamp':
            logger.info(f"Paired {len(pairs)} samples using timestamp matching")
            return pairs
    
    # Strategy 2: Metadata-based pairing (for remaining samples if auto)
    if pairing_strategy in ('auto', 'metadata'):
        remaining_audio = [audio_list[i] for i in range(len(audio_list)) if i not in used_audio]
        remaining_bio = [bio_list[i] for i in range(len(bio_list)) if i not in used_bio]
        
        if remaining_audio and remaining_bio:
            metadata_pairs = pair_by_metadata(remaining_audio, remaining_bio)
            for audio, bio, score in metadata_pairs:
                audio_idx = audio_list.index(audio)
                bio_idx = bio_list.index(bio)
                if audio_idx not in used_audio and bio_idx not in used_bio:
                    used_audio.add(audio_idx)
                    used_bio.add(bio_idx)
                    pairs.append(PairedSample(
                        audio=audio,
                        biometric=bio,
                        pairing_method='metadata',
                        pairing_score=score
                    ))
    
    # Strategy 3: Index-based pairing (fallback for remaining samples)
    if pairing_strategy in ('auto', 'index'):
        remaining_audio = [audio_list[i] for i in range(len(audio_list)) if i not in used_audio]
        remaining_bio = [bio_list[i] for i in range(len(bio_list)) if i not in used_bio]
        
        if remaining_audio and remaining_bio:
            index_pairs = pair_by_index(remaining_audio, remaining_bio)
            for audio, bio, score in index_pairs:
                audio_idx = audio_list.index(audio)
                bio_idx = bio_list.index(bio)
                if audio_idx not in used_audio and bio_idx not in used_bio:
                    used_audio.add(audio_idx)
                    used_bio.add(bio_idx)
                    pairs.append(PairedSample(
                        audio=audio,
                        biometric=bio,
                        pairing_method='index',
                        pairing_score=score
                    ))
    
    logger.info(
        f"Successfully paired {len(pairs)} samples "
        f"({len(used_audio)}/{len(audio_list)} audio, {len(used_bio)}/{len(bio_list)} biometric)"
    )
    
    return pairs


def load_samples_from_directories(
    audio_dir: Path,
    hrv_dir: Optional[Path] = None,
    accel_dir: Optional[Path] = None,
    metadata_pattern: Optional[str] = None
) -> Tuple[List[AudioSample], List[BiometricSample]]:
    """
    Load audio and biometric samples from directories.
    
    Args:
        audio_dir: Directory containing audio files
        hrv_dir: Directory containing HRV JSON files
        accel_dir: Directory containing accelerometer JSON files
        metadata_pattern: Pattern for metadata files (e.g., '{stem}.json')
        
    Returns:
        Tuple of (audio_samples, biometric_samples)
    """
    audio_samples = []
    biometric_samples = []
    
    # Load audio samples
    if audio_dir.exists():
        audio_files = list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.mp3'))
        for audio_file in audio_files:
            metadata_file = None
            if metadata_pattern:
                metadata_file = audio_dir / metadata_pattern.format(stem=audio_file.stem)
            
            sample = AudioSample.from_file(audio_file, metadata_file)
            audio_samples.append(sample)
    
    # Load biometric samples
    # Match HRV and accelerometer files by name
    hrv_files = {}
    accel_files = {}
    
    if hrv_dir and hrv_dir.exists():
        for hrv_file in hrv_dir.glob('*.json'):
            hrv_files[hrv_file.stem] = hrv_file
    
    if accel_dir and accel_dir.exists():
        for accel_file in accel_dir.glob('*.json'):
            accel_files[accel_file.stem] = accel_file
    
    # Create biometric samples
    all_bio_stems = set(hrv_files.keys()) | set(accel_files.keys())
    for stem in all_bio_stems:
        bio_sample = BiometricSample.from_files(
            hrv_file=hrv_files.get(stem),
            accel_file=accel_files.get(stem)
        )
        biometric_samples.append(bio_sample)
    
    return audio_samples, biometric_samples

