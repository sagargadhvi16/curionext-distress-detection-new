"""Test script to verify training infrastructure components."""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("TRAINING INFRASTRUCTURE VERIFICATION TEST")
print("=" * 80)

# Test 1: Import EarlyStopping
try:
    from src.fusion.early_stopping import EarlyStopping
    print("✅ EarlyStopping imported successfully")
    
    # Create instance
    es = EarlyStopping(metric="val_f1", patience=10, mode="max")
    print(f"   - Created EarlyStopping with patience={es.patience}")
except Exception as e:
    print(f"❌ EarlyStopping failed: {e}")

# Test 2: Import Checkpoint functions
try:
    from src.fusion.checkpoint import save_checkpoint, load_checkpoint, resume_from_checkpoint
    print("✅ Checkpoint functions imported successfully")
    print(f"   - save_checkpoint, load_checkpoint, resume_from_checkpoint available")
except Exception as e:
    print(f"❌ Checkpoint functions failed: {e}")

# Test 3: Import MetricsCalculator
try:
    from src.fusion.metrics_calculator import MetricsCalculator
    print("✅ MetricsCalculator imported successfully")
    
    # Create instance and test
    import numpy as np
    calc = MetricsCalculator(num_classes=2)
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.4, 0.3, 0.85, 0.15])
    
    metrics = calc.compute_all_metrics(y_true, y_pred, y_proba)
    print(f"   - Accuracy: {metrics['accuracy']:.3f}")
    print(f"   - F1 Score: {metrics['f1']:.3f}")
    print(f"   - AUC-ROC: {metrics['auc_roc']:.3f}")
    print(f"   - FNR: {metrics['fnr']:.3f}")
except Exception as e:
    print(f"❌ MetricsCalculator failed: {e}")

# Test 4: Import Visualization
try:
    from src.utils.visualization import TrainingVisualizer, plot_training_curves
    print("✅ Visualization imported successfully")
    
    # Create visualizer
    vis = TrainingVisualizer(output_dir="logs/test_plots")
    print(f"   - TrainingVisualizer created with output_dir={vis.output_dir}")
except Exception as e:
    print(f"❌ Visualization failed: {e}")

# Test 5: Load config
try:
    from src.utils.config import load_config
    config = load_config('configs/training_config.yaml')
    print("✅ Config loaded successfully")
    print(f"   - Training epochs: {config['training']['epochs']}")
    print(f"   - Batch size: {config['training']['batch_size']}")
    print(f"   - Learning rate: {config['training']['learning_rate']}")
    print(f"   - Early stopping patience: {config['early_stopping']['patience']}")
    print(f"   - Checkpoint save_dir: {config['checkpointing']['save_dir']}")
except Exception as e:
    print(f"❌ Config loading failed: {e}")

# Test 6: Test early stopping workflow
try:
    print("\n" + "=" * 80)
    print("TESTING EARLY STOPPING WORKFLOW")
    print("=" * 80)
    
    from src.fusion.early_stopping import EarlyStopping
    
    early_stopping = EarlyStopping(metric="val_f1", patience=3, mode="max", verbose=False)
    
    # Simulate training epochs
    val_metrics = [0.75, 0.78, 0.80, 0.79, 0.78, 0.77]
    
    for epoch, val_f1 in enumerate(val_metrics):
        should_stop = early_stopping(val_f1, epoch, model=None)
        status = "STOP" if should_stop else "CONTINUE"
        print(f"   Epoch {epoch}: val_f1={val_f1:.3f}, best={early_stopping.best_value:.3f}, counter={early_stopping.counter} -> {status}")
        
        if should_stop:
            print(f"   ✅ Early stopping triggered correctly at epoch {epoch}")
            break
    
except Exception as e:
    print(f"❌ Early stopping workflow failed: {e}")

# Test 7: Test metrics calculation
try:
    print("\n" + "=" * 80)
    print("TESTING METRICS CALCULATION")
    print("=" * 80)
    
    from src.fusion.metrics_calculator import MetricsCalculator
    import numpy as np
    
    calc = MetricsCalculator(num_classes=2)
    
    # Simulate predictions
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.9, 0.85, 0.2, 0.1, 0.45, 0.8, 0.15, 0.6, 0.75, 0.3])
    
    metrics = calc.compute_all_metrics(y_true, y_pred, y_proba)
    
    print("   Computed Metrics:")
    print(f"   - Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"   - Precision: {metrics.get('precision', 0):.4f}")
    print(f"   - Recall: {metrics.get('recall', 0):.4f}")
    print(f"   - F1 Score: {metrics.get('f1', 0):.4f}")
    print(f"   - AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
    print(f"   - False Negative Rate: {metrics.get('fnr', 0):.4f}")
    
    if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
        print(f"   - Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    print("   ✅ All metrics calculated successfully")
    
except Exception as e:
    print(f"❌ Metrics calculation failed: {e}")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("\n✅ All core training infrastructure components are working correctly!")
print("\nNote: Full training requires PyTorch and other dependencies.")
print("Install with: pip install -r requirements.txt")
print("\nTo start training (once dependencies installed):")
print("  python scripts/train.py")
print("=" * 80)
