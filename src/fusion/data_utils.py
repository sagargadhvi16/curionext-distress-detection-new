"""Data utilities for multimodal training: splitting and loading."""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime

from src.fusion.pairing import PairedSample, load_samples_from_directories, pair_multimodal_samples
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_data_splits(
    paired_samples: List[PairedSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: str = 'label',
    random_seed: int = 42
) -> Tuple[List[PairedSample], List[PairedSample], List[PairedSample]]:
    """
    Create train/validation/test splits with stratification.
    
    Args:
        paired_samples: List of paired samples
        train_ratio: Proportion of training data
        val_ratio: Proportion of validation data
        test_ratio: Proportion of test data
        stratify_by: Field to stratify by ('label', 'none')
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    logger.info(f"Creating data splits: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
    
    # Extract labels for stratification
    if stratify_by == 'label':
        labels = []
        for sample in paired_samples:
            # Get label from audio or biometric
            label = sample.audio.label or sample.biometric.label
            labels.append(label if label else 'unknown')
        labels = np.array(labels)
    else:
        labels = None
    
    indices = np.arange(len(paired_samples))
    
    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(1 - train_ratio),
        stratify=labels if labels is not None else None,
        random_state=random_seed,
        shuffle=True
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    temp_labels = labels[temp_indices] if labels is not None else None
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_size),
        stratify=temp_labels if temp_labels is not None else None,
        random_state=random_seed,
        shuffle=True
    )
    
    train_samples = [paired_samples[i] for i in train_indices]
    val_samples = [paired_samples[i] for i in val_indices]
    test_samples = [paired_samples[i] for i in test_indices]
    
    logger.info(
        f"Split complete: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )
    
    # Log label distribution
    if labels is not None:
        for split_name, split_samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            split_labels = [
                s.audio.label or s.biometric.label or 'unknown' 
                for s in split_samples
            ]
            unique, counts = np.unique(split_labels, return_counts=True)
            logger.info(f"{split_name} label distribution: {dict(zip(unique, counts))}")
    
    return train_samples, val_samples, test_samples

