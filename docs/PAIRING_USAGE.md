# Multimodal Sample Pairing Guide

This guide explains how to use the `pair_multimodal_samples` function to pair audio and biometric samples for multimodal fusion.

## Overview

The pairing module (`src/fusion/pairing.py`) provides functionality to intelligently pair audio samples with biometric samples (HRV + accelerometer data) based on:

1. **Timestamp matching** - Pairs samples that were recorded at similar times
2. **Metadata matching** - Pairs samples with similar labels and durations
3. **Index pairing** - Fallback sequential pairing

## Basic Usage

### Import the Function

```python
from src.fusion.pairing import pair_multimodal_samples, AudioSample, BiometricSample
```

### Create Sample Objects

```python
# Audio sample
audio_sample = AudioSample(
    file_path=Path("data/synthetic/audio/distress_001.wav"),
    timestamp=datetime(2024, 12, 16, 10, 30, 0),
    label="distress",
    duration=3.0
)

# Biometric sample (HRV + Accelerometer)
bio_sample = BiometricSample(
    hrv_file_path=Path("data/synthetic/biometric/hrv/distress_001.json"),
    accel_file_path=Path("data/synthetic/biometric/accelerometer/distress_001.json"),
    timestamp=datetime(2024, 12, 16, 10, 30, 1),  # 1 second later
    label="distress",
    duration=3.0
)
```

### Pair Samples

```python
# Pair two lists of samples
audio_list = [audio_sample1, audio_sample2, ...]
bio_list = [bio_sample1, bio_sample2, ...]

paired_samples = pair_multimodal_samples(
    audio_list=audio_list,
    bio_list=bio_list,
    pairing_strategy='auto',  # or 'timestamp', 'metadata', 'index'
    time_tolerance_seconds=5.0  # Max time difference for timestamp pairing
)
```

## Pairing Strategies

### 1. Auto Strategy (Recommended)

Automatically tries multiple strategies in order:
1. Timestamp-based pairing
2. Metadata-based pairing (for remaining samples)
3. Index-based pairing (for remaining samples)

```python
paired = pair_multimodal_samples(
    audio_list,
    bio_list,
    pairing_strategy='auto'
)
```

### 2. Timestamp Strategy

Only pairs samples with timestamps within the tolerance window:

```python
paired = pair_multimodal_samples(
    audio_list,
    bio_list,
    pairing_strategy='timestamp',
    time_tolerance_seconds=5.0  # 5 second tolerance
)
```

### 3. Metadata Strategy

Pairs samples based on labels and duration similarity:

```python
paired = pair_multimodal_samples(
    audio_list,
    bio_list,
    pairing_strategy='metadata'
)
```

### 4. Index Strategy

Simple sequential pairing (lowest quality):

```python
paired = pair_multimodal_samples(
    audio_list,
    bio_list,
    pairing_strategy='index'
)
```

## Loading Samples from Directories

Use the helper function to automatically load samples from directories:

```python
from src.fusion.pairing import load_samples_from_directories
from pathlib import Path

audio_dir = Path("data/synthetic/audio")
hrv_dir = Path("data/synthetic/biometric/hrv")
accel_dir = Path("data/synthetic/biometric/accelerometer")

audio_samples, biometric_samples = load_samples_from_directories(
    audio_dir=audio_dir,
    hrv_dir=hrv_dir,
    accel_dir=accel_dir
)

# Now pair them
paired = pair_multimodal_samples(audio_samples, biometric_samples)
```

## Working with Paired Samples

After pairing, you get a list of `PairedSample` objects:

```python
for paired in paired_samples:
    # Access audio sample
    audio_path = paired.audio.file_path
    audio_label = paired.audio.label
    audio_duration = paired.audio.duration
    
    # Access biometric sample
    hrv_data = paired.biometric.hrv_data
    accel_data = paired.biometric.accel_data
    bio_label = paired.biometric.label
    
    # Pairing information
    method = paired.pairing_method  # 'timestamp', 'metadata', or 'index'
    score = paired.pairing_score    # Confidence score (0.0 to 1.0)
```

## Complete Example

```python
from pathlib import Path
from datetime import datetime
from src.fusion.pairing import (
    pair_multimodal_samples,
    load_samples_from_directories
)

# Load samples from directories
audio_samples, biometric_samples = load_samples_from_directories(
    audio_dir=Path("data/synthetic/audio"),
    hrv_dir=Path("data/synthetic/biometric/hrv"),
    accel_dir=Path("data/synthetic/biometric/accelerometer")
)

print(f"Loaded {len(audio_samples)} audio and {len(biometric_samples)} biometric samples")

# Pair samples
paired_samples = pair_multimodal_samples(
    audio_list=audio_samples,
    bio_list=biometric_samples,
    pairing_strategy='auto',
    time_tolerance_seconds=5.0
)

print(f"Successfully paired {len(paired_samples)} samples")

# Use paired samples
for i, paired in enumerate(paired_samples):
    print(f"Pair {i+1}:")
    print(f"  Audio: {paired.audio.file_path.name}")
    print(f"  Label: {paired.audio.label}")
    print(f"  Pairing method: {paired.pairing_method}")
    print(f"  Confidence: {paired.pairing_score:.2f}")
```

## Data Format Requirements

### Audio Samples

Audio samples should be WAV or MP3 files. Optional metadata JSON can accompany them:

```json
{
  "timestamp": "2024-12-16T10:30:00Z",
  "duration_sec": 3.0,
  "label": "distress",
  "sample_rate": 16000
}
```

### Biometric Samples

HRV data (JSON format):
```json
{
  "rr_intervals": [800, 820, 795, 810, ...],
  "duration_sec": 60,
  "timestamp": "2024-12-16T10:30:00Z",
  "label": "distress"
}
```

Accelerometer data (JSON format):
```json
{
  "x": [0.1, 0.2, 0.15, ...],
  "y": [0.05, 0.08, 0.06, ...],
  "z": [9.8, 9.79, 9.81, ...],
  "sampling_rate": 50,
  "duration_sec": 60,
  "timestamp": "2024-12-16T10:30:00Z",
  "label": "distress"
}
```

## Testing

Run the test script to verify pairing functionality:

```bash
# Test with manual examples
python scripts/test_pairing.py --mode manual

# Test with synthetic data (if available)
python scripts/test_pairing.py --mode synthetic

# Test both
python scripts/test_pairing.py --mode both
```

## Notes

- Timestamp pairing has highest priority and confidence when timestamps are available
- Metadata pairing (by label) is used when timestamps are missing or samples don't match
- Index pairing is a last resort fallback
- Each sample is only paired once (no duplicates)
- Pairing score indicates confidence: 1.0 = high confidence, 0.5 = low confidence

