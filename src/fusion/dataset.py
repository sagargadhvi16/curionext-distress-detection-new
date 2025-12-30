"""PyTorch Dataset and DataLoader for multimodal data."""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import json

from src.fusion.pairing import PairedSample, AudioSample, BiometricSample
from src.audio.preprocessing import AudioPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal distress detection.
    
    Handles:
    - Audio file loading and preprocessing
    - Biometric data loading (HRV + accelerometer)
    - Context metadata encoding
    - Label extraction
    """
    
    def __init__(
        self,
        paired_samples: List[PairedSample],
        audio_preprocessor: Optional[AudioPreprocessor] = None,
        context_fields: Optional[List[str]] = None
    ):
        """
        Initialize multimodal dataset.
        
        Args:
            paired_samples: List of paired audio-biometric samples
            audio_preprocessor: Audio preprocessing pipeline
            context_fields: List of context metadata fields to encode
        """
        self.paired_samples = paired_samples
        self.audio_preprocessor = audio_preprocessor or AudioPreprocessor()
        self.context_fields = context_fields or ['time_of_day', 'location', 'child_age', 'activity_level']
        
        logger.info(f"Initialized dataset with {len(paired_samples)} samples")
    
    def __len__(self) -> int:
        return len(self.paired_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single multimodal sample.
        
        Returns:
            Dictionary with keys:
            - audio: preprocessed audio tensor
            - biometric: biometric features tensor
            - context: context metadata tensor
            - label: distress label (0 or 1)
            - severity: distress severity score (0-10)
            - distress_type: distress type class (0-4)
        """
        paired = self.paired_samples[idx]
        
        # Load and preprocess audio
        audio_path = paired.audio.file_path
        try:
            audio = self.audio_preprocessor.process(str(audio_path))
            audio_tensor = torch.FloatTensor(audio)
        except Exception as e:
            logger.warning(f"Error loading audio {audio_path}: {e}, using zeros")
            audio_tensor = torch.zeros(16000 * 3)  # 3 seconds default
        
        # Load biometric data
        biometric_features = self._load_biometric_features(paired.biometric)
        biometric_tensor = torch.FloatTensor(biometric_features)
        
        # Encode context metadata
        context_tensor = self._encode_context(paired)
        
        # Extract labels
        label = self._extract_label(paired)
        severity = self._extract_severity(paired)
        distress_type = self._extract_distress_type(paired)
        
        return {
            'audio': audio_tensor,
            'biometric': biometric_tensor,
            'context': context_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'severity': torch.tensor(severity, dtype=torch.float32),
            'distress_type': torch.tensor(distress_type, dtype=torch.long),
            'sample_id': idx
        }
    
    def _load_biometric_features(self, bio_sample: BiometricSample) -> np.ndarray:
        """
        Load and extract biometric features from HRV and accelerometer data.
        
        Returns:
            Feature vector (flattened)
        """
        features = []
        
        # HRV features
        if bio_sample.hrv_data:
            hrv = bio_sample.hrv_data
            rr_intervals = hrv.get('rr_intervals', [])
            if len(rr_intervals) > 0:
                rr_array = np.array(rr_intervals)
                features.extend([
                    np.mean(rr_array),
                    np.std(rr_array),
                    np.min(rr_array),
                    np.max(rr_array),
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Accelerometer features
        if bio_sample.accel_data:
            accel = bio_sample.accel_data
            for axis in ['x', 'y', 'z']:
                axis_data = accel.get(axis, [])
                if len(axis_data) > 0:
                    axis_array = np.array(axis_data)
                    features.extend([
                        np.mean(axis_array),
                        np.std(axis_array),
                    ])
                else:
                    features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Pad or truncate to fixed size (e.g., 20 features)
        target_size = 20
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _encode_context(self, paired: PairedSample) -> torch.Tensor:
        """
        Encode context metadata.
        
        For now, returns a simple encoding. Can be expanded with actual metadata.
        """
        # Placeholder: create simple context encoding
        # In real scenario, extract from metadata
        context_values = [
            0.5,  # time_of_day (normalized 0-1)
            0.0,  # location (one-hot encoded)
            0.5,  # child_age (normalized 0-1)
            0.5,  # activity_level (normalized 0-1)
        ]
        
        # Pad to 64 dimensions (will be properly encoded by ContextEncoder)
        context_array = np.array(context_values, dtype=np.float32)
        
        return torch.FloatTensor(context_array)
    
    def _extract_label(self, paired: PairedSample) -> int:
        """Extract binary distress label (0=normal, 1=distress)."""
        label_str = paired.audio.label or paired.biometric.label
        if label_str:
            label_lower = label_str.lower()
            if 'distress' in label_lower or label_lower == '1':
                return 1
        return 0
    
    def _extract_severity(self, paired: PairedSample) -> float:
        """Extract distress severity score (0-10)."""
        # Check metadata
        metadata = paired.metadata or {}
        severity = metadata.get('severity')
        
        if severity is not None:
            return float(np.clip(severity, 0.0, 10.0))
        
        # Infer from label
        label = self._extract_label(paired)
        return 5.0 if label == 1 else 0.0
    
    def _extract_distress_type(self, paired: PairedSample) -> int:
        """
        Extract distress type class.
        
        Classes:
        0: crying
        1: fear
        2: pain
        3: verbal_abuse
        4: emergency
        """
        metadata = paired.metadata or {}
        distress_type = metadata.get('distress_type', 'crying')
        
        type_map = {
            'crying': 0,
            'fear': 1,
            'pain': 2,
            'verbal_abuse': 3,
            'emergency': 4
        }
        
        return type_map.get(distress_type.lower(), 0)


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching multimodal samples.
    
    Handles variable-length audio sequences by padding.
    """
    # Separate fields
    audio_list = [item['audio'] for item in batch]
    biometric_list = [item['biometric'] for item in batch]
    context_list = [item['context'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    severities = torch.stack([item['severity'] for item in batch])
    distress_types = torch.stack([item['distress_type'] for item in batch])
    
    # Pad audio sequences to same length
    max_audio_len = max(audio.shape[0] for audio in audio_list)
    audio_padded = []
    for audio in audio_list:
        pad_len = max_audio_len - audio.shape[0]
        if pad_len > 0:
            audio = torch.cat([audio, torch.zeros(pad_len)])
        audio_padded.append(audio)
    audio_batch = torch.stack(audio_padded)
    
    # Stack biometric and context (should be fixed size)
    biometric_batch = torch.stack(biometric_list)
    context_batch = torch.stack(context_list)
    
    return {
        'audio': audio_batch,
        'biometric': biometric_batch,
        'context': context_batch,
        'label': labels,
        'severity': severities,
        'distress_type': distress_types
    }


class MultiModalDataLoader:
    """
    Wrapper class for creating DataLoaders with proper configuration.
    """
    
    @staticmethod
    def create_dataloader(
        dataset: MultimodalDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False
    ) -> DataLoader:
        """
        Create a DataLoader for multimodal data.
        
        Args:
            dataset: MultimodalDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer
            
        Returns:
            Configured DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_multimodal_batch,
            drop_last=False
        )

