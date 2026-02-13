# Fusion Layer & Model Implementation

This document describes the implementation of the multimodal fusion layer and end-to-end distress detection model for Intern 3.

## Overview

The fusion module combines audio, biometric, and context signals to detect child distress with three tasks:
1. Binary distress detection (distress/no_distress)
2. Severity regression (0-10 scale)
3. Distress type classification (5 classes: crying, fear, pain, verbal_abuse, emergency)

## Components Implemented

### 1. Data Utilities (`src/fusion/data_utils.py`)

**Function: `create_data_splits()`**
- Creates train/validation/test splits with stratification
- Supports 70/15/15 split (configurable)
- Stratifies by label to ensure balanced splits
- Uses sklearn's `train_test_split` with random seed for reproducibility

### 2. Dataset & DataLoader (`src/fusion/dataset.py`)

**Class: `MultimodalDataset`**
- PyTorch Dataset for multimodal data
- Handles audio file loading and preprocessing
- Loads biometric data (HRV + accelerometer) from JSON files
- Encodes context metadata
- Extracts labels, severity, and distress type

**Function: `collate_multimodal_batch()`**
- Custom collate function for batching
- Handles variable-length audio sequences with padding
- Stacks fixed-size biometric and context features

**Class: `MultiModalDataLoader`**
- Wrapper for creating DataLoaders
- Configures batch size, shuffling, workers, and pin_memory

### 3. Context Encoder (`src/fusion/context_encoder.py`)

**Class: `ContextEncoder`**
- Encodes contextual metadata into 64-dimensional embeddings
- Handles:
  - Time of day (continuous, normalized)
  - Location (categorical, embedding)
  - Child age (continuous, normalized)
  - Activity level (continuous, normalized)
- Uses separate encoders for each field, then projects to 64-dim output

### 4. Late Fusion Layer (`src/fusion/late_fusion.py`)

**Class: `LateFusionLayer`**
- Concatenation-based fusion
- Inputs: Audio (256-dim) + Biometric (256-dim) + Context (64-dim) = 576-dim
- Fusion layers: [512, 256] with batch normalization and dropout
- Output: 256-dimensional fused embedding

### 5. Attention Fusion (`src/fusion/attention_fusion.py`)

**Class: `AttentionFusion`**
- Cross-modal attention for adaptive weighting
- Dynamically weights audio, biometric, and context embeddings
- Computes attention weights (3 values for 3 modalities)
- Projects modalities to common dimension, applies attention, outputs fused embedding

### 6. Multi-Task Classifier (`src/fusion/classifier.py`)

**Class: `MultiTaskClassifier`**
- Three output heads:
  1. **Binary distress head**: 2-class classification (distress/no_distress)
  2. **Severity head**: Regression output (0-10 scale, uses Sigmoid then scales)
  3. **Type head**: 5-class classification (crying, fear, pain, verbal_abuse, emergency)
- Shared feature extraction layers
- Batch normalization and dropout for regularization

### 7. End-to-End Model (`src/fusion/model.py`)

**Class: `DistressDetectionModel`**
- Integrates all components:
  1. Audio encoder (accepts pre-initialized or creates placeholder)
  2. Biometric encoder (accepts pre-initialized or creates placeholder)
  3. Context encoder
  4. Fusion (late fusion or attention fusion, configurable)
  5. Multi-task classifier
- Forward pass through complete pipeline
- Predict method for inference

## Usage Example

```python
from src.fusion.pairing import load_samples_from_directories, pair_multimodal_samples
from src.fusion.data_utils import create_data_splits
from src.fusion.dataset import MultimodalDataset, MultiModalDataLoader
from src.fusion.model import DistressDetectionModel
from src.audio.encoder import AudioEncoder
from src.biometric.encoder import BiometricEncoder

# 1. Load and pair samples
audio_samples, bio_samples = load_samples_from_directories(
    audio_dir=Path("data/synthetic/audio"),
    hrv_dir=Path("data/synthetic/biometric/hrv"),
    accel_dir=Path("data/synthetic/biometric/accelerometer")
)
paired_samples = pair_multimodal_samples(audio_samples, bio_samples)

# 2. Create data splits
train_samples, val_samples, test_samples = create_data_splits(
    paired_samples,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# 3. Create datasets and dataloaders
train_dataset = MultimodalDataset(train_samples)
train_loader = MultiModalDataLoader.create_dataloader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# 4. Initialize model
audio_encoder = AudioEncoder(output_dim=256)
bio_encoder = BiometricEncoder(input_dim=20, embedding_dim=256)

model = DistressDetectionModel(
    audio_encoder=audio_encoder,
    biometric_encoder=bio_encoder,
    use_attention_fusion=False,  # or True for attention fusion
    fusion_hidden_dims=[512, 256],
    dropout=0.4
)

# 5. Forward pass
for batch in train_loader:
    outputs = model(
        audio_features=batch['audio'],  # After audio encoding
        biometric_features=batch['biometric'],  # After biometric encoding
        context_features=batch['context']
    )
    
    # outputs contains:
    # - distress_logits: (batch_size, 2)
    # - severity: (batch_size, 1) in range [0, 10]
    # - type_logits: (batch_size, 5)
```

## File Structure

```
src/fusion/
├── __init__.py              # Module exports
├── pairing.py               # Sample pairing (from previous task)
├── data_utils.py            # Data splitting utilities
├── dataset.py               # PyTorch Dataset and DataLoader
├── context_encoder.py       # Context metadata encoder
├── late_fusion.py           # Concatenation-based fusion
├── attention_fusion.py      # Attention-based fusion
├── classifier.py            # Multi-task classifier
└── model.py                 # End-to-end model

scripts/
└── test_data_pipeline.py    # Integration tests
```

## Integration Tests

Run the integration test script to verify all components:

```bash
python scripts/test_data_pipeline.py
```

Tests:
1. Data loading and pairing
2. Data splitting with stratification
3. Dataset and DataLoader creation
4. Model forward pass

## Notes

1. **Audio Encoder Integration**: The model accepts pre-initialized audio encoders. Use `AudioEncoder` from `src.audio.encoder` which expects spectrograms (batch, 1, freq_bins, time_steps) as input.

2. **Biometric Encoder Integration**: Similarly, use `BiometricEncoder` from `src.biometric.encoder` which expects biometric feature sequences (batch, seq_len, feature_dim).

3. **Context Encoding**: The `ContextEncoder` handles metadata encoding. The dataset extracts context from paired samples, but in production, you'd pass actual metadata (time_of_day, location, child_age, activity_level).

4. **Fusion Options**: Choose between `LateFusionLayer` (concatenation) or `AttentionFusion` (adaptive weighting) via the `use_attention_fusion` parameter.

5. **Multi-Task Learning**: The classifier outputs three predictions simultaneously. Use appropriate loss functions:
   - Binary cross-entropy for distress detection
   - MSE/MAE for severity regression
   - Cross-entropy for distress type classification

## Deliverables Checklist

✅ `create_data_splits()` function with stratified sampling  
✅ `MultiModalDataLoader` class with `__getitem__` and `collate_fn`  
✅ `test_data_pipeline.py` with integration tests  
✅ `ContextEncoder` class embedding metadata as 64-dim vector  
✅ `LateFusionLayer` class concatenating embeddings (256+256+64=576-dim)  
✅ `AttentionFusion` module dynamically weighting modalities  
✅ Regularized fusion layer preventing overfitting (batch norm + dropout)  
✅ `MultiTaskClassifier` with 3 output heads  
✅ Severity estimation head in classifier (0-10 scale)  
✅ 5-class distress type classifier  
✅ `DistressDetectionModel` class with complete forward pass  

All deliverables have been implemented and are ready for integration and testing!

