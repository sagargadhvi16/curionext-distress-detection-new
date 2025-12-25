"""Test script for multimodal sample pairing."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fusion.pairing import (
    pair_multimodal_samples,
    AudioSample,
    BiometricSample,
    load_samples_from_directories
)
from src.utils.logger import setup_logger
import logging


def test_pairing_with_synthetic_data():
    """Test pairing function with synthetic data structure."""
    logger = setup_logger(__name__, level=logging.INFO)
    logger.info("Testing multimodal sample pairing...")
    
    # Define directories
    audio_dir = project_root / "data/synthetic/audio"
    hrv_dir = project_root / "data/synthetic/biometric/hrv"
    accel_dir = project_root / "data/synthetic/biometric/accelerometer"
    
    # Create directories if they don't exist
    audio_dir.mkdir(parents=True, exist_ok=True)
    hrv_dir.mkdir(parents=True, exist_ok=True)
    accel_dir.mkdir(parents=True, exist_ok=True)
    
    # Load samples
    logger.info("Loading samples from directories...")
    audio_samples, biometric_samples = load_samples_from_directories(
        audio_dir=audio_dir,
        hrv_dir=hrv_dir,
        accel_dir=accel_dir
    )
    
    logger.info(f"Loaded {len(audio_samples)} audio samples and {len(biometric_samples)} biometric samples")
    
    if len(audio_samples) == 0 or len(biometric_samples) == 0:
        logger.warning(
            "No samples found. Please generate synthetic data first:\n"
            "  python scripts/generate_synthetic_audio.py\n"
            "  python scripts/generate_synthetic_biometric.py"
        )
        return
    
    # Test pairing
    logger.info("Pairing samples...")
    paired_samples = pair_multimodal_samples(
        audio_list=audio_samples,
        bio_list=biometric_samples,
        pairing_strategy='auto'
    )
    
    logger.info(f"Successfully paired {len(paired_samples)} samples")
    
    # Display pairing results
    print("\n" + "="*80)
    print("PAIRING RESULTS")
    print("="*80)
    
    for i, paired in enumerate(paired_samples[:10]):  # Show first 10
        print(f"\nPair {i+1}:")
        print(f"  Audio: {paired.audio.file_path.name}")
        print(f"  Biometric: HRV={paired.biometric.hrv_file_path.name if paired.biometric.hrv_file_path else 'N/A'}, "
              f"Accel={paired.biometric.accel_file_path.name if paired.biometric.accel_file_path else 'N/A'}")
        print(f"  Method: {paired.pairing_method}")
        print(f"  Score: {paired.pairing_score:.2f}")
    
    if len(paired_samples) > 10:
        print(f"\n... and {len(paired_samples) - 10} more pairs")
    
    # Statistics
    print("\n" + "="*80)
    print("PAIRING STATISTICS")
    print("="*80)
    
    methods = {}
    for paired in paired_samples:
        method = paired.pairing_method
        methods[method] = methods.get(method, 0) + 1
    
    for method, count in methods.items():
        print(f"  {method}: {count} pairs")
    
    avg_score = sum(p.pairing_score for p in paired_samples) / len(paired_samples) if paired_samples else 0
    print(f"\n  Average pairing score: {avg_score:.2f}")
    print("="*80 + "\n")
    
    return paired_samples


def test_manual_pairing():
    """Test pairing with manually created samples."""
    logger = setup_logger(__name__, level=logging.INFO)
    logger.info("Testing manual sample pairing...")
    
    from datetime import datetime, timedelta
    
    # Create test samples with timestamps
    base_time = datetime.now()
    
    audio_samples = [
        AudioSample(
            file_path=Path("test_audio_1.wav"),
            timestamp=base_time,
            label="distress",
            duration=3.0
        ),
        AudioSample(
            file_path=Path("test_audio_2.wav"),
            timestamp=base_time + timedelta(seconds=5),
            label="normal",
            duration=3.0
        ),
    ]
    
    biometric_samples = [
        BiometricSample(
            hrv_file_path=Path("test_hrv_1.json"),
            timestamp=base_time + timedelta(seconds=1),  # 1 second difference
            label="distress",
            duration=3.0
        ),
        BiometricSample(
            hrv_file_path=Path("test_hrv_2.json"),
            timestamp=base_time + timedelta(seconds=6),  # 1 second difference
            label="normal",
            duration=3.0
        ),
    ]
    
    # Test timestamp pairing
    paired = pair_multimodal_samples(
        audio_samples,
        biometric_samples,
        pairing_strategy='timestamp',
        time_tolerance_seconds=5.0
    )
    
    print(f"\n[OK] Manual pairing test: {len(paired)} pairs created")
    for i, p in enumerate(paired):
        print(f"  Pair {i+1}: {p.audio.file_path.name} <-> {p.biometric.hrv_file_path.name} "
              f"(method: {p.pairing_method}, score: {p.pairing_score:.2f})")
    
    return paired


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test multimodal sample pairing')
    parser.add_argument(
        '--mode',
        choices=['synthetic', 'manual', 'both'],
        default='both',
        help='Test mode (default: both)'
    )
    
    args = parser.parse_args()
    
    if args.mode in ('synthetic', 'both'):
        test_pairing_with_synthetic_data()
    
    if args.mode in ('manual', 'both'):
        test_manual_pairing()
    
    print("\n[SUCCESS] Pairing tests completed!")

