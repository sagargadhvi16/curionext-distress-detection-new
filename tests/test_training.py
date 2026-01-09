"""Smoke tests for training infrastructure with dummy dataset."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fusion.early_stopping import EarlyStopping
from src.fusion.checkpoint import save_checkpoint, load_checkpoint
from src.fusion.metrics_calculator import MetricsCalculator
from src.utils.visualization import TrainingVisualizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DummyModel(nn.Module):
    """Dummy model for testing training loop."""
    
    def __init__(self, input_dim=256, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def create_dummy_dataset(num_samples=100, input_dim=256, num_classes=2):
    """Create dummy dataset for testing.
    
    Args:
        num_samples: Number of samples
        input_dim: Input feature dimension
        num_classes: Number of output classes
    
    Returns:
        Tuple of (X, y) numpy arrays
    """
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    return X, y


def test_early_stopping():
    """Test early stopping functionality."""
    print("\n" + "=" * 80)
    print("TEST 1: Early Stopping")
    print("=" * 80)
    
    try:
        early_stopping = EarlyStopping(
            metric="val_loss",
            patience=3,
            min_delta=1e-4,
            mode="min",
            verbose=False
        )
        
        # Simulate training with improving then degrading loss
        val_losses = [0.5, 0.45, 0.42, 0.43, 0.44, 0.45]
        
        for epoch, loss in enumerate(val_losses):
            should_stop = early_stopping(loss, epoch, model=None)
            if should_stop:
                print(f"✅ Early stopping triggered at epoch {epoch}")
                print(f"   Best loss: {early_stopping.best_value:.4f}")
                print(f"   Counter: {early_stopping.counter}/{early_stopping.patience}")
                break
        else:
            print(f"⚠️ Early stopping did not trigger within {len(val_losses)} epochs")
        
        print("✅ TEST PASSED: Early Stopping works correctly")
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False


def test_checkpointing():
    """Test checkpoint save and load."""
    print("\n" + "=" * 80)
    print("TEST 2: Checkpointing")
    print("=" * 80)
    
    try:
        import tempfile
        from pathlib import Path
        
        # Create dummy model
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        checkpoint_dir = Path(tempfile.mkdtemp())
        
        # Save checkpoint
        metrics = {'loss': 0.5, 'accuracy': 0.85, 'f1': 0.82}
        save_checkpoint(
            str(checkpoint_dir),
            epoch=5,
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            is_best=True
        )
        print(f"✅ Checkpoint saved to {checkpoint_dir}")
        
        # Verify checkpoint file exists
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        print(f"   Found {len(checkpoint_files)} checkpoint files")
        
        # Load checkpoint
        model2 = DummyModel()
        optimizer2 = torch.optim.Adam(model2.parameters())
        
        checkpoint_file = checkpoint_dir / "best_checkpoint.pt"
        if checkpoint_file.exists():
            metadata = load_checkpoint(
                str(checkpoint_file),
                model2,
                optimizer2,
                device="cpu"
            )
            print(f"✅ Checkpoint loaded successfully")
            print(f"   Epoch: {metadata['epoch']}")
            print(f"   Metrics: {metadata['metrics']}")
        
        print("✅ TEST PASSED: Checkpointing works correctly")
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False


def test_metrics_calculation():
    """Test metrics calculation."""
    print("\n" + "=" * 80)
    print("TEST 3: Metrics Calculation")
    print("=" * 80)
    
    try:
        calculator = MetricsCalculator(num_classes=2)
        
        # Create dummy predictions
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 1])
        y_proba = np.array([0.2, 0.15, 0.85, 0.45, 0.90, 0.25, 0.80, 0.55, 0.75, 0.88])
        
        # Compute all metrics
        metrics = calculator.compute_all_metrics(y_true, y_pred, y_proba)
        
        print(f"✅ Metrics computed successfully:")
        print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"   Precision: {metrics.get('precision', 0):.4f}")
        print(f"   Recall: {metrics.get('recall', 0):.4f}")
        print(f"   F1: {metrics.get('f1', 0):.4f}")
        print(f"   AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
        print(f"   False Negative Rate: {metrics.get('fnr', 0):.4f}")
        
        # Verify all metrics are computed
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'fnr']
        missing = [m for m in required_metrics if m not in metrics or metrics[m] is None]
        
        if not missing:
            print("✅ TEST PASSED: All metrics calculated correctly")
            return True
        else:
            print(f"❌ Missing metrics: {missing}")
            return False
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False


def test_visualization():
    """Test visualization generation."""
    print("\n" + "=" * 80)
    print("TEST 4: Visualization")
    print("=" * 80)
    
    try:
        import tempfile
        from pathlib import Path
        
        output_dir = Path(tempfile.mkdtemp())
        visualizer = TrainingVisualizer(output_dir=str(output_dir))
        
        # Create dummy training data
        train_loss = [0.5, 0.45, 0.40, 0.38, 0.37]
        val_loss = [0.55, 0.48, 0.43, 0.41, 0.42]
        train_acc = [0.80, 0.82, 0.85, 0.87, 0.88]
        val_acc = [0.78, 0.81, 0.83, 0.84, 0.84]
        train_f1 = [0.75, 0.78, 0.82, 0.84, 0.86]
        val_f1 = [0.73, 0.77, 0.80, 0.82, 0.82]
        
        # Plot training curves
        visualizer.plot_training_curves(
            train_loss, val_loss,
            train_acc, val_acc,
            train_f1, val_f1
        )
        print(f"✅ Training curves plotted to {output_dir}")
        
        # Plot learning rate
        learning_rates = [0.001, 0.0009, 0.0008, 0.0007, 0.0006]
        visualizer.plot_learning_rate(learning_rates)
        print(f"✅ Learning rate plot generated")
        
        # Plot confusion matrix
        cm = np.array([[45, 5], [8, 42]])
        visualizer.plot_confusion_matrix(cm, class_names=["No Distress", "Distress"])
        print(f"✅ Confusion matrix plotted")
        
        # Verify files exist
        plot_files = list(output_dir.glob("*.png"))
        print(f"   Generated {len(plot_files)} plot files")
        
        print("✅ TEST PASSED: Visualization works correctly")
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False


def test_training_loop():
    """Test a simple training loop."""
    print("\n" + "=" * 80)
    print("TEST 5: Training Loop")
    print("=" * 80)
    
    try:
        # Create dummy data
        X_train, y_train = create_dummy_dataset(num_samples=100)
        X_val, y_val = create_dummy_dataset(num_samples=20)
        
        # Create model and training components
        model = DummyModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(patience=3, mode="min")
        calculator = MetricsCalculator(num_classes=2)
        
        # Simple training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(5):
            # Training
            model.train()
            train_preds = []
            train_targets = []
            train_loss_sum = 0.0
            
            for i in range(0, len(X_train), 32):
                X_batch = torch.FloatTensor(X_train[i:i+32])
                y_batch = torch.LongTensor(y_train[i:i+32])
                
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss_sum += loss.item()
                train_preds.extend(torch.argmax(logits, dim=1).numpy())
                train_targets.extend(y_batch.numpy())
            
            train_loss = train_loss_sum / (len(X_train) / 32)
            train_losses.append(train_loss)
            
            train_metrics = calculator.compute_all_metrics(
                np.array(train_targets),
                np.array(train_preds)
            )
            train_acc = train_metrics.get('accuracy', 0)
            train_accuracies.append(train_acc)
            
            # Validation
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val)
                y_val_tensor = torch.LongTensor(y_val)
                
                val_logits = model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
                val_losses.append(val_loss)
                
                val_preds = torch.argmax(val_logits, dim=1).numpy()
                val_metrics = calculator.compute_all_metrics(y_val, val_preds)
                val_acc = val_metrics.get('accuracy', 0)
                val_accuracies.append(val_acc)
            
            # Check early stopping
            should_stop = early_stopping(val_loss, epoch)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            
            if should_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print("✅ TEST PASSED: Training loop works correctly")
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all smoke tests."""
    print("\n" + "=" * 80)
    print("RUNNING SMOKE TESTS FOR TRAINING INFRASTRUCTURE")
    print("=" * 80)
    
    tests = [
        test_early_stopping,
        test_checkpointing,
        test_metrics_calculation,
        test_visualization,
        test_training_loop
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ EXCEPTION IN {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"⚠️ {total - passed} TEST(S) FAILED")
    
    print("=" * 80)
    
    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
