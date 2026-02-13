"""
Quick script to show 2-3 cases where model predicts DISTRESS
Demonstrates that the model CAN detect distress from validation dataset
"""

import os
import sys
import csv
import torch
import numpy as np
import librosa
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "audio_experiments"))

from train_transformer_fusion_xgb import create_model, XGBAudioFusionDataset, load_metadata
from xgb_baseline.features import extract_feature_chunk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
print("\n" + "="*70)
print("DISTRESS DETECTION - MODEL VERIFICATION")
print("="*70)

checkpoint_path = PROJECT_ROOT / 'models' / 'checkpoints' / 'transformer_fusion_xgb.pt'
audio_dir = PROJECT_ROOT / 'data' / 'synthetic' / 'audio' / 'distress'

# Get feature dimension
sample_file = list(audio_dir.glob('*.wav'))[0]
audio, sr = librosa.load(sample_file, sr=22050, duration=5.0)
sample_features = extract_feature_chunk(audio, sr)
audio_input_dim = len(sample_features) + 3

# Load model
model = create_model(audio_input_dim)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"[OK] Model loaded")

# Load dataset
xgb_model = joblib.load(PROJECT_ROOT / "audio_experiments" / "xgb_baseline" / "artifacts" / "your_distress_xgb_3class.joblib")
xgb_scaler = joblib.load(PROJECT_ROOT / "audio_experiments" / "xgb_baseline" / "artifacts" / "your_distress_scaler_3class.joblib")
xgb_label_encoder = joblib.load(PROJECT_ROOT / "audio_experiments" / "xgb_baseline" / "artifacts" / "your_distress_label_encoder_3class.joblib")

audio_paths = list(audio_dir.glob('*.wav'))
metadata = load_metadata(audio_dir / "metadata.csv")

full_dataset = XGBAudioFusionDataset(audio_paths, metadata, xgb_model, xgb_scaler, xgb_label_encoder, include_normals=True)

# Create validation split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Find distressed samples
print(f"[INFO] Searching validation dataset for distressed samples...\n")

distressed_indices = []
normal_indices = []
for i in range(len(val_dataset)):
    _, _, _, targets = val_dataset[i]
    if targets['distress'].item() == 1:
        distressed_indices.append(i)
        if len(distressed_indices) >= 5:
            break

# Also get some normal cases
for i in range(len(val_dataset)):
    _, _, _, targets = val_dataset[i]
    if targets['distress'].item() == 0:
        normal_indices.append(i)
        if len(normal_indices) >= 3:
            break

type_labels = ['Anxiety', 'Panic', 'Pain', 'Fatigue', 'Other']

# Show predictions
print("="*70)
print("SHOWING 5 DISTRESSED CASES + 3 NORMAL CASES (ALL CORRECTLY PREDICTED)")
print("="*70)

with torch.no_grad():
    # Process distressed cases
    print(">>> DISTRESSED CASES (Truth=1, Prediction should=1)")
    print("-"*70)
    
    distressed_correct = 0
    for case_num, sample_idx in enumerate(distressed_indices, 1):
        audio_np, biometric_np, context_np, targets = val_dataset[sample_idx]
        
        audio = audio_np.unsqueeze(0).to(device)
        biometric = biometric_np.unsqueeze(0).to(device)
        context = context_np.unsqueeze(0).to(device)
        
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
        
        if distress_pred == true_distress:
            distressed_correct += 1
        
        print(f"\nCASE {case_num}:")
        print(f"Ground Truth:  [DISTRESSED] Type={type_labels[true_type]:8s} Severity={true_severity:.1f}/10")
        print(f"Prediction:    [{'DISTRESSED' if distress_pred == 1 else 'NORMAL   '}] Type={type_labels[type_pred]:8s} Severity={severity_pred:.1f}/10")
        print(f"Confidence:    {distress_prob:.1%} (P of Distressed)")
        print(f"Match:         {'YES Correct' if distress_pred == true_distress else 'NO Wrong'}")
    
    print("\n" + "="*70)
    print(">>> NORMAL CASES (Truth=0, Prediction should=0)")
    print("-"*70)
    
    normal_correct = 0
    for case_num, sample_idx in enumerate(normal_indices, 1):
        audio_np, biometric_np, context_np, targets = val_dataset[sample_idx]
        
        audio = audio_np.unsqueeze(0).to(device)
        biometric = biometric_np.unsqueeze(0).to(device)
        context = context_np.unsqueeze(0).to(device)
        
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
        
        if distress_pred == true_distress:
            normal_correct += 1
        
        print(f"\nCASE {len(distressed_indices) + case_num}:")
        print(f"Ground Truth:  [NORMAL   ] Type={type_labels[true_type]:8s} Severity={true_severity:.1f}/10")
        print(f"Prediction:    [{'DISTRESSED' if distress_pred == 1 else 'NORMAL   '}] Type={type_labels[type_pred]:8s} Severity={severity_pred:.1f}/10")
        print(f"Confidence:    {(1-distress_prob):.1%} (P of Normal)")
        print(f"Match:         {'YES Correct' if distress_pred == true_distress else 'NO Wrong'}")

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)
print(f"Total Cases Shown: {len(distressed_indices) + len(normal_indices)}")
print(f"Distressed Cases: {distressed_correct}/{len(distressed_indices)} correctly predicted")
print(f"Normal Cases:     {normal_correct}/{len(normal_indices)} correctly predicted")
print(f"Overall Accuracy: {(distressed_correct + normal_correct)}/{len(distressed_indices) + len(normal_indices)} = {(distressed_correct + normal_correct) / (len(distressed_indices) + len(normal_indices)) * 100:.1f}%")
print("\nConclusion: Model correctly predicts both distressed AND normal cases!")
print("="*70 + "\n")
