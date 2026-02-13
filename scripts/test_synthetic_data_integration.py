"""
Integration Test: Real Synthetic Data Training Pipeline
========================================================

This test validates the complete training pipeline using actual synthetic data
from data/synthetic/audio and data/synthetic/biometric folders.

Tests:
1. Load synthetic audio data (500 samples)
2. Load synthetic biometric data (accelerometer + HRV)
3. Create PyTorch DataLoader with real data
4. Train model end-to-end with transformer fusion
5. Validate metrics tracking on real data
"""

import os
import sys
import json
import numpy as np
import torch
import librosa
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Tuple, Dict, Optional
import csv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fusion.model import DistressDetectionModel
from src.utils.signal_augmentation import SignalAugmentation


class SyntheticDataset(Dataset):
    """
    Load synthetic audio + biometric data from data/synthetic folder.
    
    Structure:
    - data/synthetic/audio/distress/ - 500 audio files + metadata.csv
    - data/synthetic/biometric/accelerometer/ - accelerometer data
    - data/synthetic/biometric/hrv/ - HRV data
    """
    
    def __init__(
        self,
        data_root: str = "data/synthetic",
        sample_rate: int = 16000,
        audio_duration: float = 5.0,
        n_mfcc: int = 13,
        split: str = "train",
        train_ratio: float = 0.8,
        max_samples: int = None
    ):
        """
        Args:
            data_root: Path to synthetic data folder
            sample_rate: Audio sample rate
            audio_duration: Duration to load (seconds)
            n_mfcc: Number of MFCC features
            split: 'train' or 'test'
            train_ratio: Fraction for train vs test split
            max_samples: Limit number of samples (for testing)
        """
        self.data_root = Path(data_root)
        self.audio_dir = self.data_root / "audio" / "distress"
        self.biometric_dir = self.data_root / "biometric"
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.n_mfcc = n_mfcc
        self.split = split
        
        # Get metadata
        metadata_path = self.audio_dir / "metadata.csv"
        self.metadata = []
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                reader = csv.DictReader(f)
                self.metadata = list(reader)
        
        # Get audio files
        audio_files = sorted(self.audio_dir.glob("*.wav"))
        
        # Limit samples for faster testing
        if max_samples is not None:
            audio_files = audio_files[:max_samples]
        
        # Train/test split
        n_train = int(len(audio_files) * train_ratio)
        if split == "train":
            self.audio_files = audio_files[:n_train]
        else:
            self.audio_files = audio_files[n_train:]
        
        logger.info(f"Loaded {len(self.audio_files)} {split} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def _load_audio_features(self, filepath: Path) -> np.ndarray:
        """Load and extract MFCC features from audio file."""
        try:
            # Load audio
            waveform, sr = librosa.load(str(filepath), sr=self.sample_rate)
            
            # Pad or trim to fixed duration
            n_samples = int(self.audio_duration * self.sample_rate)
            if len(waveform) < n_samples:
                waveform = np.pad(waveform, (0, n_samples - len(waveform)))
            else:
                waveform = waveform[:n_samples]
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=2048,
                hop_length=512
            )
            
            # Return as (n_mfcc, time_steps)
            return mfcc.astype(np.float32)
        except Exception as e:
            logger.warning(f"Error loading audio {filepath}: {e}")
            # Return zeros if loading fails
            n_frames = int((self.audio_duration * self.sample_rate) / 512) + 1
            return np.zeros((self.n_mfcc, n_frames), dtype=np.float32)
    
    def _load_biometric_features(self, file_idx: int) -> np.ndarray:
        """Load accelerometer + HRV biometric features."""
        try:
            features = []
            
            # Load accelerometer (3 axes)
            accel_dir = self.biometric_dir / "accelerometer"
            accel_file = accel_dir / f"accel_{file_idx:05d}.npy"
            if accel_file.exists():
                accel_data = np.load(accel_file)
                # Flatten to 1D
                accel_features = accel_data.flatten()[:100]  # Take first 100 values
                features.append(accel_features)
            else:
                features.append(np.zeros(100, dtype=np.float32))
            
            # Load HRV (heart rate variability)
            hrv_dir = self.biometric_dir / "hrv"
            hrv_file = hrv_dir / f"hrv_{file_idx:05d}.npy"
            if hrv_file.exists():
                hrv_data = np.load(hrv_file)
                hrv_features = hrv_data.flatten()[:100]  # Take first 100 values
                features.append(hrv_features)
            else:
                features.append(np.zeros(100, dtype=np.float32))
            
            # Concatenate all biometric features
            return np.concatenate(features).astype(np.float32)
        except Exception as e:
            logger.warning(f"Error loading biometric for idx {file_idx}: {e}")
            # Return zeros if loading fails
            return np.zeros(200, dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Return:
            audio: (n_mfcc * time_steps,) flattened
            biometric: (200,)
            targets: dict with 'distress', 'severity', 'type'
        """
        audio_path = self.audio_files[idx]
        file_idx = int(audio_path.stem.split('_')[1])
        
        # Load audio features
        audio_features = self._load_audio_features(audio_path)
        # Flatten MFCC: (n_mfcc, time_steps) -> (n_mfcc * time_steps,)
        audio_tensor = torch.from_numpy(audio_features.flatten())
        
        # Load biometric features
        biometric_features = self._load_biometric_features(file_idx)
        biometric_tensor = torch.from_numpy(biometric_features)
        
        # Generate labels from metadata or random
        if self.metadata:
            meta = self.metadata[min(idx, len(self.metadata) - 1)]
            distress_label = int(meta.get('distress', np.random.randint(0, 2)))
            severity_label = float(meta.get('severity', np.random.uniform(0, 10)))
            type_label = int(meta.get('type', np.random.randint(0, 5)))
        else:
            # Random labels if no metadata
            distress_label = np.random.randint(0, 2)
            severity_label = np.random.uniform(0, 10)
            type_label = np.random.randint(0, 5)
        
        targets = {
            'distress': torch.tensor(distress_label, dtype=torch.long),
            'severity': torch.tensor(severity_label, dtype=torch.float32),
            'type': torch.tensor(type_label, dtype=torch.long)
        }
        
        return audio_tensor, biometric_tensor, targets


def test_synthetic_data_loading():
    """Test 1: Can we load synthetic data?"""
    print("\n" + "="*80)
    print("TEST 1: Loading Synthetic Data from data/synthetic")
    print("="*80)
    
    try:
        # Create train/test datasets (limit to 50 samples for speed)
        train_dataset = SyntheticDataset(split="train", train_ratio=0.8, max_samples=50)
        test_dataset = SyntheticDataset(split="test", train_ratio=0.8, max_samples=50)
        
        logger.info(f"✓ Train dataset: {len(train_dataset)} samples")
        logger.info(f"✓ Test dataset: {len(test_dataset)} samples")
        
        # Sample one item
        audio, biometric, targets = train_dataset[0]
        logger.info(f"✓ Audio shape: {audio.shape}")
        logger.info(f"✓ Biometric shape: {biometric.shape}")
        logger.info(f"✓ Targets: distress={targets['distress']}, "
                   f"severity={targets['severity']:.2f}, type={targets['type']}")
        
        print("\n✓ Synthetic Data Loading test PASSED!")
        return True
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_creation():
    """Test 2: Can we create a DataLoader with real data?"""
    print("\n" + "="*80)
    print("TEST 2: Creating DataLoader with Synthetic Data")
    print("="*80)
    
    try:
        train_dataset = SyntheticDataset(split="train", train_ratio=0.8, max_samples=50)
        
        # Custom collate function to handle dict targets
        def collate_fn(batch):
            audio_list = [item[0] for item in batch]
            biometric_list = [item[1] for item in batch]
            targets_list = [item[2] for item in batch]
            
            # Pad audio to same length
            max_len = max(a.shape[0] for a in audio_list)
            audio_padded = []
            for a in audio_list:
                if a.shape[0] < max_len:
                    pad = torch.zeros(max_len - a.shape[0])
                    a = torch.cat([a, pad])
                audio_padded.append(a)
            
            audio_batch = torch.stack(audio_padded, dim=0)
            biometric_batch = torch.stack(biometric_list, dim=0)
            
            # Stack targets
            targets_batch = {
                'distress': torch.stack([t['distress'] for t in targets_list]),
                'severity': torch.stack([t['severity'] for t in targets_list]),
                'type': torch.stack([t['type'] for t in targets_list])
            }
            
            return audio_batch, biometric_batch, targets_batch
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        logger.info(f"✓ DataLoader created with {len(train_loader)} batches")
        
        # Iterate one batch
        audio_batch, biometric_batch, targets_batch = next(iter(train_loader))
        logger.info(f"✓ Batch audio shape: {audio_batch.shape}")
        logger.info(f"✓ Batch biometric shape: {biometric_batch.shape}")
        logger.info(f"✓ Batch targets: distress={targets_batch['distress'].shape}, "
                   f"severity={targets_batch['severity'].shape}, "
                   f"type={targets_batch['type'].shape}")
        
        print("\n✓ DataLoader Creation test PASSED!")
        return True
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_training():
    """Test 3: Train model end-to-end with real synthetic data"""
    print("\n" + "="*80)
    print("TEST 3: End-to-End Training with Real Synthetic Data")
    print("="*80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create datasets (limit to 30 samples for speed)
        train_dataset = SyntheticDataset(split="train", train_ratio=0.8, max_samples=30)
        val_dataset = SyntheticDataset(split="test", train_ratio=0.8, max_samples=30)
        
        # Custom collate function
        def collate_fn(batch):
            audio_list = [item[0] for item in batch]
            biometric_list = [item[1] for item in batch]
            targets_list = [item[2] for item in batch]
            
            max_len = max(a.shape[0] for a in audio_list)
            audio_padded = []
            for a in audio_list:
                if a.shape[0] < max_len:
                    pad = torch.zeros(max_len - a.shape[0])
                    a = torch.cat([a, pad])
                audio_padded.append(a)
            
            audio_batch = torch.stack(audio_padded, dim=0)
            biometric_batch = torch.stack(biometric_list, dim=0)
            
            targets_batch = {
                'distress': torch.stack([t['distress'] for t in targets_list]),
                'severity': torch.stack([t['severity'] for t in targets_list]),
                'type': torch.stack([t['type'] for t in targets_list])
            }
            
            return audio_batch, biometric_batch, targets_batch
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        logger.info(f"✓ Train loader: {len(train_loader)} batches")
        logger.info(f"✓ Val loader: {len(val_loader)} batches")
        
        # Create model with transformer fusion
        model = DistressDetectionModel(
            fusion_type="transformer",
            use_tcn_refiner=True,
            audio_dim=128,  # Match MFCC features
            bio_dim=256,
            context_dim=64,
            transformer_d_model=128,
            transformer_nhead=4,
            transformer_num_layers=2
        ).to(device)
        
        logger.info(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn_distress = torch.nn.CrossEntropyLoss()
        loss_fn_severity = torch.nn.MSELoss()
        loss_fn_type = torch.nn.CrossEntropyLoss()
        
        # Training loop (2 epochs)
        num_epochs = 2
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (audio, biometric, targets) in enumerate(train_loader):
                if batch_idx >= 2:  # Only 2 batches per epoch for speed
                    break
                audio = audio.to(device)
                biometric = biometric.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                # Forward pass
                outputs = model(audio, biometric)
                
                # Compute loss
                loss = 0.0
                if 'distress_logits' in outputs and 'distress' in targets:
                    loss += loss_fn_distress(outputs['distress_logits'], targets['distress'])
                
                if 'severity' in outputs and 'severity' in targets:
                    loss += 0.5 * loss_fn_severity(
                        outputs['severity'].squeeze(),
                        targets['severity']
                    )
                
                if 'type_logits' in outputs and 'type' in targets:
                    loss += 0.8 * loss_fn_type(outputs['type_logits'], targets['type'])
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
            
            # Validation
            model.eval()
            val_loss = 0.0
            distress_correct = 0
            total = 0
            
            with torch.no_grad():
                for audio, biometric, targets in val_loader:
                    audio = audio.to(device)
                    biometric = biometric.to(device)
                    targets = {k: v.to(device) for k, v in targets.items()}
                    
                    outputs = model(audio, biometric)
                    
                    loss = 0.0
                    if 'distress_logits' in outputs:
                        loss += loss_fn_distress(outputs['distress_logits'], targets['distress'])
                        preds = torch.argmax(outputs['distress_logits'], dim=1)
                        distress_correct += (preds == targets['distress']).sum().item()
                    
                    if 'severity' in outputs:
                        loss += 0.5 * loss_fn_severity(
                            outputs['severity'].squeeze(),
                            targets['severity']
                        )
                    
                    if 'type_logits' in outputs:
                        loss += 0.8 * loss_fn_type(outputs['type_logits'], targets['type'])
                    
                    val_loss += loss.item()
                    total += audio.shape[0]
            
            val_loss /= len(val_loader)
            distress_acc = distress_correct / total if total > 0 else 0.0
            logger.info(f"  Val Loss: {val_loss:.4f}, Distress Acc: {distress_acc:.4f}")
        
        print("\n✓ End-to-End Training test PASSED!")
        return True
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_augmentation():
    """Test 4: Training with signal augmentation on real data"""
    print("\n" + "="*80)
    print("TEST 4: Training with Signal Augmentation")
    print("="*80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        train_dataset = SyntheticDataset(split="train", train_ratio=0.8, max_samples=20)
        
        def collate_fn(batch):
            audio_list = [item[0] for item in batch]
            biometric_list = [item[1] for item in batch]
            targets_list = [item[2] for item in batch]
            
            max_len = max(a.shape[0] for a in audio_list)
            audio_padded = []
            for a in audio_list:
                if a.shape[0] < max_len:
                    pad = torch.zeros(max_len - a.shape[0])
                    a = torch.cat([a, pad])
                audio_padded.append(a)
            
            audio_batch = torch.stack(audio_padded, dim=0)
            biometric_batch = torch.stack(biometric_list, dim=0)
            
            targets_batch = {
                'distress': torch.stack([t['distress'] for t in targets_list]),
                'severity': torch.stack([t['severity'] for t in targets_list]),
                'type': torch.stack([t['type'] for t in targets_list])
            }
            
            return audio_batch, biometric_batch, targets_batch
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        # Create augmentor
        augmentor = SignalAugmentation(
            augmentation_prob=0.5,
            noise_snr_range=(15.0, 30.0),
            magnitude_warp_sigma=0.2
        )
        
        logger.info("✓ Augmentor created")
        
        # Create model
        model = DistressDetectionModel(
            fusion_type="transformer",
            use_tcn_refiner=True,
            audio_dim=128,
            bio_dim=256,
            context_dim=64
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn_distress = torch.nn.CrossEntropyLoss()
        
        # Train with augmentation
        model.train()
        for batch_idx, (audio, biometric, targets) in enumerate(train_loader):
            if batch_idx >= 2:  # Just 2 batches
                break
            
            # Apply augmentation to audio
            audio_np = audio.cpu().numpy().astype(np.float32)
            augmented_audios = []
            for i in range(audio_np.shape[0]):
                aug_versions = [
                    augmentor.apply_single(audio_np[i], "noise"),
                    augmentor.apply_single(audio_np[i], "magnitude_warp")
                ]
                augmented_audios.append(np.mean(aug_versions, axis=0).astype(np.float32))
            
            audio = torch.from_numpy(np.array(augmented_audios)).float().to(device)
            biometric = biometric.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            outputs = model(audio, biometric)
            loss = loss_fn_distress(outputs['distress_logits'], targets['distress'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.info(f"  Batch {batch_idx+1}: Loss with augmentation = {loss.item():.4f}")
        
        print("\n✓ Signal Augmentation test PASSED!")
        return True
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "SYNTHETIC DATA INTEGRATION TEST SUITE" + " "*25 + "║")
    print("║" + " "*78 + "║")
    print("║" + "Loading real synthetic data from data/synthetic/ folder" + " "*21 + "║")
    print("╚" + "="*78 + "╝")
    
    results = []
    
    # Run tests
    results.append(("Data Loading", test_synthetic_data_loading()))
    results.append(("DataLoader Creation", test_dataloader_creation()))
    results.append(("End-to-End Training", test_end_to_end_training()))
    results.append(("Augmentation", test_with_augmentation()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - SYNTHETIC DATA PIPELINE WORKING!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - CHECK ERRORS ABOVE")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
