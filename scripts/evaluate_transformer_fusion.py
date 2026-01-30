"""
Comprehensive evaluation script for transformer fusion model.
Generates confusion matrices, metrics reports, ROC curves, and attention visualizations.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score
)
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "audio_experiments"))

from torch.utils.data import DataLoader
from train_transformer_fusion_xgb import (
    XGBAudioFusionDataset, create_model,
    SYNTH_AUDIO_DIR, METADATA_PATH,
    XGB_MODEL_PATH, XGB_SCALER_PATH, XGB_ENCODER_PATH,
    load_metadata
)
import joblib

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_trained_model(checkpoint_path, audio_input_dim):
    """Load the trained model from checkpoint."""
    model = create_model(audio_input_dim)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✅ Loaded model from {checkpoint_path}")
    return model

def evaluate_model(model, dataloader):
    """Run inference and collect predictions."""
    all_distress_preds = []
    all_distress_probs = []
    all_distress_targets = []
    all_type_preds = []
    all_type_targets = []
    all_severity_preds = []
    all_severity_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            audio = batch['audio'].to(device)
            biometric = batch['biometric'].to(device)
            context = batch['context'].to(device)
            
            distress_target = batch['distress_label'].to(device)
            severity_target = batch['severity'].to(device)
            type_target = batch['type_label'].to(device)
            
            # Forward pass
            outputs = model(audio, biometric, context)
            distress_logits = outputs['distress_logits']
            severity = outputs['severity']
            type_logits = outputs['type_logits']
            
            # Distress detection (2-class classification)
            distress_probs = torch.softmax(distress_logits, dim=1)[:, 1].cpu().numpy()  # Probability of class 1 (distressed)
            distress_preds = torch.argmax(distress_logits, dim=1).cpu().numpy()
            
            # Type classification
            type_preds = torch.argmax(type_logits, dim=1).cpu().numpy()
            
            # Severity regression
            severity_preds = severity.squeeze().cpu().numpy()
            
            # Collect results
            all_distress_preds.append(distress_preds)
            all_distress_probs.append(distress_probs)
            all_distress_targets.append(distress_target.cpu().numpy())
            all_type_preds.append(type_preds)
            all_type_targets.append(type_target.cpu().numpy())
            all_severity_preds.append(severity_preds)
            all_severity_targets.append(severity_target.cpu().numpy())
    
    # Ensure all arrays are 1D
    results = {
        'distress_preds': np.concatenate(all_distress_preds).ravel(),
        'distress_probs': np.concatenate(all_distress_probs).ravel(),
        'distress_targets': np.concatenate(all_distress_targets).ravel(),
        'type_preds': np.concatenate(all_type_preds).ravel(),
        'type_targets': np.concatenate(all_type_targets).ravel(),
        'severity_preds': np.concatenate(all_severity_preds).ravel(),
        'severity_targets': np.concatenate(all_severity_targets).ravel()
    }
    
    # Debug: print shapes
    print(f"\n🔍 Result shapes:")
    for key, val in results.items():
        print(f"  {key}: {val.shape}")
    
    return results

def plot_confusion_matrices(results, output_dir):
    """Generate confusion matrices for distress and type classification."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distress confusion matrix
    cm_distress = confusion_matrix(results['distress_targets'], results['distress_preds'])
    sns.heatmap(cm_distress, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Normal', 'Distressed'], yticklabels=['Normal', 'Distressed'])
    axes[0].set_title('Distress Detection Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Type confusion matrix (only for distressed samples)
    distressed_mask = results['distress_targets'] == 1
    if distressed_mask.sum() > 0:
        cm_type = confusion_matrix(results['type_targets'][distressed_mask], 
                                   results['type_preds'][distressed_mask])
        type_labels = ['Anxiety', 'Panic', 'Pain', 'Fatigue', 'Other']
        sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                   xticklabels=type_labels, yticklabels=type_labels)
        axes[1].set_title('Distress Type Classification Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=45)
    
    plt.tight_layout()
    output_path = output_dir / 'confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved confusion matrices to {output_path}")
    plt.close()

def plot_roc_curve(results, output_dir):
    """Generate ROC curve for distress detection."""
    fpr, tpr, thresholds = roc_curve(results['distress_targets'], results['distress_probs'])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Distress Detection ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    output_path = output_dir / 'roc_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📈 Saved ROC curve to {output_path}")
    plt.close()

def plot_severity_analysis(results, output_dir):
    """Analyze severity prediction accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot: predicted vs actual
    axes[0].scatter(results['severity_targets'], results['severity_preds'], 
                   alpha=0.5, s=20)
    axes[0].plot([0, 10], [0, 10], 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Severity')
    axes[0].set_ylabel('Predicted Severity')
    axes[0].set_title('Severity Prediction: Actual vs Predicted')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Error distribution
    errors = results['severity_preds'] - results['severity_targets']
    axes[1].hist(errors, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Severity Error Distribution (MAE={np.abs(errors).mean():.3f})')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'severity_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📉 Saved severity analysis to {output_path}")
    plt.close()

def generate_metrics_report(results, output_dir):
    """Generate comprehensive metrics report."""
    report = {}
    
    # Distress detection metrics
    report['distress_detection'] = {
        'accuracy': accuracy_score(results['distress_targets'], results['distress_preds']),
        'f1_score': f1_score(results['distress_targets'], results['distress_preds']),
        'classification_report': classification_report(
            results['distress_targets'], results['distress_preds'],
            target_names=['Normal', 'Distressed'], output_dict=True
        )
    }
    
    # Type classification metrics (only distressed samples)
    distressed_mask = results['distress_targets'] == 1
    if distressed_mask.sum() > 0:
        # Get unique classes present in the data
        unique_classes = np.unique(np.concatenate([
            results['type_targets'][distressed_mask], 
            results['type_preds'][distressed_mask]
        ]))
        
        # Use only the labels that are present
        all_type_labels = ['Anxiety', 'Panic', 'Pain', 'Fatigue', 'Other']
        type_labels = [all_type_labels[i] for i in unique_classes if i < len(all_type_labels)]
        
        report['type_classification'] = {
            'accuracy': accuracy_score(results['type_targets'][distressed_mask], 
                                      results['type_preds'][distressed_mask]),
            'f1_score_macro': f1_score(results['type_targets'][distressed_mask], 
                                      results['type_preds'][distressed_mask], 
                                      average='macro'),
            'classification_report': classification_report(
                results['type_targets'][distressed_mask], 
                results['type_preds'][distressed_mask],
                labels=unique_classes.tolist(),
                target_names=type_labels, output_dict=True
            )
        }
    
    # Severity regression metrics
    mae = np.abs(results['severity_preds'] - results['severity_targets']).mean()
    mse = ((results['severity_preds'] - results['severity_targets']) ** 2).mean()
    rmse = np.sqrt(mse)
    
    report['severity_regression'] = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mean_predicted': float(results['severity_preds'].mean()),
        'mean_actual': float(results['severity_targets'].mean()),
        'std_predicted': float(results['severity_preds'].std()),
        'std_actual': float(results['severity_targets'].std())
    }
    
    # Save JSON report
    json_path = output_dir / 'metrics_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Saved metrics report to {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\n🎯 DISTRESS DETECTION:")
    print(f"  Accuracy: {report['distress_detection']['accuracy']:.3f}")
    print(f"  F1 Score: {report['distress_detection']['f1_score']:.3f}")
    
    if 'type_classification' in report:
        print(f"\n🏷️  DISTRESS TYPE CLASSIFICATION:")
        print(f"  Accuracy: {report['type_classification']['accuracy']:.3f}")
        print(f"  Macro F1: {report['type_classification']['f1_score_macro']:.3f}")
    
    print(f"\n📊 SEVERITY REGRESSION:")
    print(f"  MAE: {report['severity_regression']['mae']:.3f}")
    print(f"  RMSE: {report['severity_regression']['rmse']:.3f}")
    print("="*60 + "\n")
    
    return report

def main():
    # Paths
    checkpoint_path = PROJECT_ROOT / 'models' / 'checkpoints' / 'transformer_fusion_xgb.pt'
    output_dir = PROJECT_ROOT / 'evaluation_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("TRANSFORMER FUSION MODEL EVALUATION")
    print("="*60 + "\n")
    
    # Load dataset first to get feature dimensions
    print("📦 Loading validation dataset...")
    
    # Load XGBoost artifacts
    xgb_model = joblib.load(XGB_MODEL_PATH)
    scaler = joblib.load(XGB_SCALER_PATH)
    label_encoder = joblib.load(XGB_ENCODER_PATH)
    
    # Load audio files and metadata
    audio_paths = list(SYNTH_AUDIO_DIR.glob('*.wav'))
    metadata = load_metadata(METADATA_PATH)
    
    if not audio_paths:
        raise RuntimeError(f"No audio files found in {SYNTH_AUDIO_DIR}")
    
    full_dataset = XGBAudioFusionDataset(audio_paths, metadata, xgb_model, scaler, label_encoder, include_normals=True)
    
    # Load model with correct dimensions
    model = load_trained_model(checkpoint_path, full_dataset.feature_dim)
    
    # Use same split as training (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    def collate_fn(batch):
        audio, biometric, context, targets = zip(*batch)
        return {
            'audio': torch.stack(audio, dim=0).float(),
            'biometric': torch.stack(biometric, dim=0).float(),
            'context': torch.stack(context, dim=0).float(),
            'distress_label': torch.stack([t["distress"] for t in targets]),
            'severity': torch.stack([t["severity"] for t in targets]),
            'type_label': torch.stack([t["type"] for t in targets]),
        }
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Run evaluation
    print("\n🔬 Running inference on validation set...")
    results = evaluate_model(model, val_loader)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_confusion_matrices(results, output_dir)
    plot_roc_curve(results, output_dir)
    plot_severity_analysis(results, output_dir)
    
    # Generate metrics report
    print("\n📈 Generating metrics report...")
    report = generate_metrics_report(results, output_dir)
    
    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")
    print(f"   - Confusion matrices: confusion_matrices.png")
    print(f"   - ROC curve: roc_curve.png")
    print(f"   - Severity analysis: severity_analysis.png")
    print(f"   - Metrics report: metrics_report.json\n")

if __name__ == '__main__':
    main()
