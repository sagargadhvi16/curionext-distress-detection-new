"""
Inference demo script for transformer fusion model.
Tests the model on individual audio files and displays predictions.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import librosa
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "audio_experiments"))

from train_transformer_fusion_xgb import create_model
from xgb_baseline.features import extract_feature_chunk

# XGBoost artifacts paths
XGB_MODEL_PATH = PROJECT_ROOT / "audio_experiments" / "xgb_baseline" / "artifacts" / "your_distress_xgb_3class.joblib"
XGB_SCALER_PATH = PROJECT_ROOT / "audio_experiments" / "xgb_baseline" / "artifacts" / "your_distress_scaler_3class.joblib"
XGB_ENCODER_PATH = PROJECT_ROOT / "audio_experiments" / "xgb_baseline" / "artifacts" / "your_distress_label_encoder_3class.joblib"

# Load XGBoost models
xgb_model = joblib.load(XGB_MODEL_PATH)
xgb_scaler = joblib.load(XGB_SCALER_PATH)
xgb_label_encoder = joblib.load(XGB_ENCODER_PATH)

def predict_xgb(features):
    """Run XGBoost prediction on audio features."""
    feat_scaled = xgb_scaler.transform([features])
    pred_enc = xgb_model.predict(feat_scaled)[0]
    pred_label = xgb_label_encoder.inverse_transform([pred_enc])[0]
    confidence = xgb_model.predict_proba(feat_scaled).max()
    return pred_label, confidence

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DistressInference:
    """Inference wrapper for distress detection model."""
    
    def __init__(self, checkpoint_path, audio_input_dim):
        """Load trained model."""
        self.model = create_model(audio_input_dim)
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        print(f"[OK] Loaded model from {checkpoint_path}")
        
        self.type_labels = ['Anxiety', 'Panic', 'Pain', 'Fatigue', 'Other']
    
    def generate_synthetic_biometric(self, distress_type_idx, severity):
        """Generate synthetic biometric data based on distress characteristics."""
        np.random.seed(None)  # Random seed for variation
        
        # Base biometric features (200-dim)
        bio = np.random.normal(0, 1, 200)
        
        # Distress type specific patterns
        type_patterns = {
            0: np.array([1.5, 1.2, 0.8, 0.5, 0.3]),  # Anxiety: high arousal, moderate stress
            1: np.array([2.0, 1.8, 1.5, 1.0, 0.8]),  # Panic: very high arousal and stress
            2: np.array([0.8, 1.5, 1.2, 0.6, 0.4]),  # Pain: moderate arousal, high physical stress
            3: np.array([0.3, 0.5, 0.4, 1.5, 1.2]),  # Fatigue: low arousal, high tiredness
            4: np.array([1.0, 1.0, 1.0, 1.0, 1.0])   # Other: balanced
        }
        
        if distress_type_idx < 5:
            pattern = type_patterns[distress_type_idx]
            bio[:5] += pattern * (severity / 10.0)
        
        return torch.tensor(bio, dtype=torch.float32)
    
    def generate_synthetic_context(self):
        """Generate synthetic context data."""
        # Time: hour (0-23), day_of_week (0-6)
        hour = np.random.randint(0, 24)
        day = np.random.randint(0, 7)
        
        # Environmental: noise_level (0-1), ambient_temp (0-1)
        noise = np.random.uniform(0, 1)
        temp = np.random.uniform(0, 1)
        
        # Motion: is_moving (0/1), activity_level (0-1)
        moving = np.random.choice([0, 1])
        activity = np.random.uniform(0, 1)
        
        # Device: one-hot [smartphone, smartwatch, wearable]
        device = np.random.choice(3)
        device_onehot = np.eye(3)[device]
        
        # Location: one-hot [home, work, public]
        location = np.random.choice(3)
        location_onehot = np.eye(3)[location]
        
        context = np.concatenate([
            [hour/24.0, day/7.0, noise, temp, moving, activity],
            device_onehot,
            location_onehot
        ])
        
        return torch.tensor(context, dtype=torch.float32)
    
    def predict(self, audio_path, verbose=True):
        """
        Run inference on a single audio file.
        
        Args:
            audio_path: Path to audio file
            verbose: Print detailed output
            
        Returns:
            Dictionary with predictions
        """
        # 1. Load audio and extract XGBoost features
        if verbose:
            print(f"\n[AUDIO] Processing: {Path(audio_path).name}")
        
        audio, sr = librosa.load(audio_path, sr=22050, duration=5.0)
        xgb_features = extract_feature_chunk(audio, sr)
        
        # 2. Get XGBoost predictions and probabilities
        feat_scaled = xgb_scaler.transform([xgb_features])
        xgb_probs = xgb_model.predict_proba(feat_scaled)[0]  # 3-class probabilities
        pred_enc = xgb_model.predict(feat_scaled)[0]
        label = xgb_label_encoder.inverse_transform([pred_enc])[0]
        confidence = xgb_probs.max()
        
        if verbose:
            print(f"   XGBoost prediction: {label} (confidence: {confidence:.3f})")
        
        # 3. Concatenate XGBoost features + probabilities (to match training)
        combined_audio_features = np.concatenate([xgb_features, xgb_probs])
        
        # 4. Map XGBoost output to distress info
        if label == 'neutral':
            distress_type_idx = 4  # Other
            estimated_severity = 0.0
        elif label == 'high_arousal':
            distress_type_idx = 0  # Anxiety
            estimated_severity = 5.0 + np.random.uniform(0, 3)
        else:  # distress
            distress_type_idx = np.random.choice([1, 2, 3])  # Panic, Pain, or Fatigue
            estimated_severity = 6.0 + np.random.uniform(0, 4)
        
        # 5. Generate synthetic biometric and context
        audio_tensor = torch.tensor(combined_audio_features, dtype=torch.float32).unsqueeze(0).to(device)
        biometric_tensor = self.generate_synthetic_biometric(distress_type_idx, estimated_severity).unsqueeze(0).to(device)
        context_tensor = self.generate_synthetic_context().unsqueeze(0).to(device)
        
        # 6. Run transformer fusion model
        with torch.no_grad():
            outputs = self.model(audio_tensor, biometric_tensor, context_tensor)
            
            # Distress detection (2-class classification)
            distress_probs = torch.softmax(outputs['distress_logits'], dim=1)[0]
            distress_prob = distress_probs[1].item()  # Probability of class 1 (distressed)
            distress_detected = distress_probs.argmax().item() == 1
            
            # Severity prediction
            severity = outputs['severity'].item()
            severity = max(0, min(10, severity))  # Clip to [0, 10]
            
            # Type classification
            type_probs = torch.softmax(outputs['type_logits'], dim=1).cpu().numpy()[0]
            predicted_type_idx = type_probs.argmax()
            predicted_type = self.type_labels[predicted_type_idx]
        
        # 7. Display results
        if verbose:
            print("\n" + "="*50)
            print("[RESULTS] PREDICTION OUTPUT")
            print("="*50)
            print(f"\n[DISTRESS] Detected: {'YES' if distress_detected else 'NO'}")
            print(f"   Confidence: {distress_prob:.1%}")
            
            if distress_detected:
                print(f"\n[TYPE] Identified: {predicted_type}")
                print(f"   Type Probabilities:")
                for i, (label, prob) in enumerate(zip(self.type_labels, type_probs)):
                    bar = '=' * int(prob * 30)
                    print(f"      {label:10s} {bar:30s} {prob:.1%}")
                
                print(f"\n[SEVERITY] Score: {severity:.2f} / 10.0")
                severity_bar = '=' * int(severity)
                print(f"   {severity_bar:10s} ({severity:.1f}/10)")
            
            print("="*50 + "\n")
        
        return {
            'distress_detected': distress_detected,
            'distress_probability': distress_prob,
            'severity': severity,
            'distress_type': predicted_type,
            'type_probabilities': dict(zip(self.type_labels, type_probs)),
            'xgb_prediction': label,
            'xgb_confidence': confidence
        }

def predict_from_dataset(model, dataset, sample_indices):
    """Run inference on samples from training dataset using actual data."""
    results = []
    
    print(f"\n{'='*60}")
    print("PREDICTIONS ON VALIDATION DATASET SAMPLES")
    print(f"{'='*60}")
    print(f"\n[*] Using actual training data for {len(sample_indices)} samples...\n")
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(sample_indices, 1):
            # Get actual data from dataset (returns tuple: audio, biometric, context, targets_dict)
            audio_np, biometric_np, context_np, targets = dataset[sample_idx]
            
            audio = audio_np.unsqueeze(0).to(device)
            biometric = biometric_np.unsqueeze(0).to(device)
            context = context_np.unsqueeze(0).to(device)
            
            # Ground truth labels
            true_distress = targets['distress'].item()
            true_type = targets['type'].item()
            true_severity = targets['severity'].item()
            
            # Get predictions
            outputs = model(audio, biometric, context)
            
            distress_probs = torch.softmax(outputs['distress_logits'], dim=1)[0]
            distress_pred = distress_probs.argmax().item()
            distress_prob = distress_probs[1].item()
            
            type_probs = torch.softmax(outputs['type_logits'], dim=1)[0].cpu().numpy()
            type_pred = type_probs.argmax()
            
            severity_pred = outputs['severity'].squeeze().item()
            
            type_labels = ['Anxiety', 'Panic', 'Pain', 'Fatigue', 'Other']
            
            # Check if prediction is correct
            distress_correct = distress_pred == true_distress
            type_correct = type_pred == true_type
            
            print(f"[{idx}] Sample {idx}:")
            print(f"    Ground Truth: {'[DISTRESSED]' if true_distress else '[NORMAL]':12s} | Type: {type_labels[true_type]:8s} | Severity: {true_severity:.1f}")
            print(f"    Prediction:   {'[DISTRESSED]' if distress_pred else '[NORMAL]':12s} {('[OK]' if distress_correct else '[FAIL]'):6s} | P={distress_prob:.1%} | Type: {type_labels[type_pred]:8s} {('[OK]' if type_correct else '[FAIL]'):6s} | Severity: {severity_pred:.1f}")
            print()
            
            results.append({
                'true_distress': true_distress,
                'pred_distress': distress_pred,
                'distress_prob': distress_prob,
                'true_type': true_type,
                'pred_type': type_pred,
                'true_severity': true_severity,
                'pred_severity': severity_pred,
                'distress_correct': distress_correct,
                'type_correct': type_correct
            })
    
    # Summary statistics
    print(f"{'='*60}")
    print("VALIDATION DATASET PREDICTIONS SUMMARY")
    print(f"{'='*60}")
    distress_acc = np.mean([r['distress_correct'] for r in results])
    type_acc = np.mean([r['type_correct'] for r in results])
    avg_prob = np.mean([r['distress_prob'] for r in results if r['true_distress'] == 1])
    
    print(f"\n[DISTRESS] Detection Accuracy: {distress_acc:.1%}")
    print(f"[TYPE]     Classification Accuracy: {type_acc:.1%}")
    print(f"[PROB]     Average Distress Probability (actual distressed): {avg_prob:.1%}")
    print()
    
    return results


def main():
    """Demo inference on sample audio files."""
    # Get audio feature dimension: XGBoost features + 3 class probabilities
    from xgb_baseline.features import extract_feature_chunk
    import librosa
    from train_transformer_fusion_xgb import XGBAudioFusionDataset
    import csv
    
    # Load one sample to get base feature dimension
    audio_dir = PROJECT_ROOT / 'data' / 'synthetic' / 'audio' / 'distress'
    sample_file = list(audio_dir.glob('*.wav'))[0]
    audio, sr = librosa.load(sample_file, sr=22050, duration=5.0)
    sample_features = extract_feature_chunk(audio, sr)
    
    # Audio input dim = XGBoost features + 3 probabilities (from XGBoost classifier)
    audio_input_dim = len(sample_features) + 3
    
    # Load model
    checkpoint_path = PROJECT_ROOT / 'models' / 'checkpoints' / 'transformer_fusion_xgb.pt'
    inferencer_model = create_model(audio_input_dim)
    state_dict = torch.load(checkpoint_path, map_location=device)
    inferencer_model.load_state_dict(state_dict)
    inferencer_model.to(device)
    inferencer_model.eval()
    
    # Option 1: Predictions on actual validation dataset samples
    print("\n" + "="*60)
    print("MODE 1: PREDICTIONS ON VALIDATION DATASET")
    print("="*60)
    
    # Load dataset
    xgb_model = joblib.load(PROJECT_ROOT / "audio_experiments" / "xgb_baseline" / "artifacts" / "your_distress_xgb_3class.joblib")
    xgb_scaler = joblib.load(PROJECT_ROOT / "audio_experiments" / "xgb_baseline" / "artifacts" / "your_distress_scaler_3class.joblib")
    xgb_label_encoder = joblib.load(PROJECT_ROOT / "audio_experiments" / "xgb_baseline" / "artifacts" / "your_distress_label_encoder_3class.joblib")
    
    audio_paths = list(audio_dir.glob('*.wav'))
    
    # Load metadata
    metadata_path = audio_dir / "metadata.csv"
    metadata = {}
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['file']] = row
    
    full_dataset = XGBAudioFusionDataset(audio_paths, metadata, xgb_model, xgb_scaler, xgb_label_encoder, include_normals=True)
    
    # Create same split as training
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Show predictions on 5 distressed samples from validation set
    distressed_indices = []
    for i in range(len(val_dataset)):
        # Dataset returns tuple: (audio, biometric, context, targets_dict)
        _, _, _, targets = val_dataset[i]
        if targets['distress'].item() == 1:  # Distressed
            distressed_indices.append(i)
            if len(distressed_indices) >= 5:
                break
    
    dataset_results = predict_from_dataset(inferencer_model, val_dataset, distressed_indices)
    
    # Option 2: Predictions on random samples with synthetic data
    print("\n" + "="*60)
    print("MODE 2: PREDICTIONS WITH SYNTHETIC BIOMETRIC DATA")
    print("="*60)
    
    # Initialize old inferencer for synthetic predictions
    inferencer = DistressInference(checkpoint_path, audio_input_dim)
    
    # Find sample audio files
    audio_files = list(audio_dir.glob('*.wav'))[:5]  # Test on first 5 files
    
    if not audio_files:
        print("[ERROR] No audio files found in data/synthetic/audio/distress/")
        return
    
    print(f"\n[INFO] Testing on {len(audio_files)} sample audio files (with synthetic biometric)...\n")
    
    # Run inference on each file
    results = []
    for audio_file in audio_files:
        result = inferencer.predict(audio_file, verbose=True)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - SYNTHETIC BIOMETRIC RESULTS")
    print("="*60)
    distressed_count = sum(1 for r in results if r['distress_detected'])
    avg_severity = np.mean([r['severity'] for r in results if r['distress_detected']])
    
    print(f"\n[TOTAL] Samples: {len(results)}")
    print(f"[DISTRESSED] Count: {distressed_count} ({distressed_count/len(results):.1%})")
    if distressed_count > 0:
        print(f"[SEVERITY] Average: {avg_severity:.2f}/10.0")
    
    # Type distribution
    if distressed_count > 0:
        print(f"\n[TYPES] Distress type distribution:")
        from collections import Counter
        type_counts = Counter(r['distress_type'] for r in results if r['distress_detected'])
        for dtype, count in type_counts.most_common():
            print(f"   {dtype:10s}: {count}")
    
    print("\n[OK] Inference demo complete!\n")

if __name__ == '__main__':
    main()
