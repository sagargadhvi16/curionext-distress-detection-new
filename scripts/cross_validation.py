"""K-fold cross-validation for robust model evaluation."""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import argparse
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fusion.metrics_calculator import MetricsCalculator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrossValidator:
    """K-fold cross-validation evaluator."""
    
    DISTRESS_TYPES = {
        0: "crying",
        1: "fear",
        2: "pain",
        3: "discomfort",
        4: "anger"
    }
    
    def __init__(self, n_splits=5, num_classes=5, random_state=42):
        """Initialize cross-validator.
        
        Args:
            n_splits: Number of folds
            num_classes: Number of classes
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.num_classes = num_classes
        self.random_state = random_state
        
        self.metrics_calculator = MetricsCalculator(num_classes=num_classes)
        self.fold_results = []
        self.overall_results = {}
    
    def cross_validate(self, X, y, model_func=None):
        """Perform k-fold cross-validation.
        
        Args:
            X: Feature matrix (numpy array or similar)
            y: Labels (numpy array)
            model_func: Function that trains and evaluates model on fold
                       Should accept (X_train, y_train, X_val, y_val, fold_idx)
                       and return (y_pred, y_proba)
        
        Returns:
            Dictionary with cross-validation results
        """
        print(f"\n{'='*80}")
        print(f"K-FOLD CROSS-VALIDATION ({self.n_splits} FOLDS)")
        print(f"{'='*80}\n")
        
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'='*80}")
            print(f"FOLD {fold_idx + 1}/{self.n_splits}")
            print(f"{'='*80}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"Train size: {len(y_train)}, Val size: {len(y_val)}")
            print(f"Train class distribution: {np.bincount(y_train)}")
            print(f"Val class distribution: {np.bincount(y_val)}")
            
            # Train and evaluate on fold
            if model_func is not None:
                y_pred, y_proba = model_func(X_train, y_train, X_val, y_val, fold_idx)
            else:
                # Use dummy model if no function provided
                y_pred, y_proba = self._dummy_model_fold(y_train, y_val)
            
            # Compute metrics
            fold_metrics = self.metrics_calculator.compute_all_metrics(
                y_val, y_pred, y_proba
            )
            
            # Store results
            fold_result = {
                'fold': fold_idx + 1,
                'train_size': len(y_train),
                'val_size': len(y_val),
                'metrics': fold_metrics,
                'y_true': y_val.tolist() if isinstance(y_val, np.ndarray) else y_val,
                'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
            }
            
            self.fold_results.append(fold_result)
            
            # Print fold metrics
            print(f"\nFold Metrics:")
            print(f"  Accuracy: {fold_metrics.get('accuracy', 0):.4f}")
            print(f"  Precision: {fold_metrics.get('precision', 0):.4f}")
            print(f"  Recall: {fold_metrics.get('recall', 0):.4f}")
            print(f"  F1 Score: {fold_metrics.get('f1', 0):.4f}")
            print(f"  AUC-ROC: {fold_metrics.get('auc_roc', 0):.4f}")
            print(f"  False Negative Rate: {fold_metrics.get('fnr', 0):.4f}")
        
        # Compute aggregate statistics
        self._compute_aggregate_statistics()
        
        return self.fold_results
    
    def _dummy_model_fold(self, y_train, y_val):
        """Dummy model for testing (when no real model provided).
        
        Returns weighted predictions based on training distribution.
        """
        # Simple strategy: predict most common class
        most_common = np.argmax(np.bincount(y_train))
        y_pred = np.full(len(y_val), most_common)
        
        # Add some randomness to probabilities
        y_proba = np.ones((len(y_val), self.num_classes)) / self.num_classes
        
        return y_pred, y_proba
    
    def _compute_aggregate_statistics(self):
        """Compute aggregate statistics across all folds."""
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}\n")
        
        # Extract metrics from each fold
        all_metrics = {}
        
        for fold_result in self.fold_results:
            metrics = fold_result['metrics']
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float, np.integer, np.floating)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)
        
        # Compute statistics
        summary_stats = {}
        
        for metric_name, values in sorted(all_metrics.items()):
            values = np.array(values)
            
            stats = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values.tolist()
            }
            
            summary_stats[metric_name] = stats
            
            print(f"{metric_name.upper()}")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
            print()
        
        self.overall_results['summary_statistics'] = summary_stats
        self.overall_results['num_folds'] = self.n_splits
        self.overall_results['timestamp'] = datetime.now().isoformat()
    
    def save_results(self, output_dir="cv_results"):
        """Save cross-validation results.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed fold results
        fold_report_path = output_path / "cv_fold_results.json"
        
        # Convert numpy arrays to lists
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_fold_results = convert_to_serializable(self.fold_results)
        
        with open(fold_report_path, 'w') as f:
            json.dump(serializable_fold_results, f, indent=2)
        
        print(f"\n✅ Fold results saved to {fold_report_path}")
        
        # Save summary statistics
        summary_report_path = output_path / "cv_summary_statistics.json"
        serializable_overall = convert_to_serializable(self.overall_results)
        
        with open(summary_report_path, 'w') as f:
            json.dump(serializable_overall, f, indent=2)
        
        print(f"✅ Summary statistics saved to {summary_report_path}")
        
        # Save as CSV for easy viewing
        csv_path = output_path / "cv_metrics_summary.csv"
        self._save_summary_csv(csv_path)
        
        return output_path
    
    def _save_summary_csv(self, csv_path):
        """Save summary statistics as CSV."""
        rows = []
        
        summary_stats = self.overall_results.get('summary_statistics', {})
        
        for metric_name in sorted(summary_stats.keys()):
            stats = summary_stats[metric_name]
            
            row = {
                'Metric': metric_name,
                'Mean': stats['mean'],
                'Std Dev': stats['std'],
                'Min': stats['min'],
                'Max': stats['max'],
                '95% CI Lower': stats['mean'] - 1.96 * stats['std'],
                '95% CI Upper': stats['mean'] + 1.96 * stats['std']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV summary saved to {csv_path}")
    
    def get_results(self):
        """Get cross-validation results."""
        return {
            'fold_results': self.fold_results,
            'summary_statistics': self.overall_results
        }


def create_dummy_dataset(num_samples=500, num_classes=5, num_features=256):
    """Create dummy dataset for cross-validation.
    
    Args:
        num_samples: Number of samples
        num_classes: Number of classes
        num_features: Number of features
    
    Returns:
        Tuple of (X, y)
    """
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    
    return X, y


def dummy_model_function(X_train, y_train, X_val, y_val, fold_idx):
    """Dummy model training function for testing.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        fold_idx: Fold index
    
    Returns:
        Tuple of (y_pred, y_proba)
    """
    # Simple strategy: predict based on training distribution
    class_counts = np.bincount(y_train, minlength=5)
    class_probs = class_counts / class_counts.sum()
    
    # Predict most likely class with some randomness
    np.random.seed(fold_idx)  # For reproducibility
    y_pred = np.array([np.random.choice(5, p=class_probs) for _ in range(len(y_val))])
    
    # Probabilities: add noise to class probabilities
    y_proba = np.tile(class_probs, (len(y_val), 1))
    y_proba += np.random.normal(0, 0.05, y_proba.shape)
    y_proba = np.clip(y_proba, 0, 1)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    return y_pred, y_proba


def main():
    """Main cross-validation function."""
    parser = argparse.ArgumentParser(description="K-fold cross-validation")
    parser.add_argument("--num-splits", type=int, default=5,
                       help="Number of folds")
    parser.add_argument("--num-samples", type=int, default=500,
                       help="Number of samples")
    parser.add_argument("--num-classes", type=int, default=5,
                       help="Number of classes")
    parser.add_argument("--num-features", type=int, default=256,
                       help="Number of features")
    parser.add_argument("--output-dir", type=str, default="cv_results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"\n{'='*80}")
    print("K-FOLD CROSS-VALIDATION SCRIPT")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Config:")
    print(f"  Num Splits: {args.num_splits}")
    print(f"  Num Samples: {args.num_samples}")
    print(f"  Num Classes: {args.num_classes}")
    print(f"  Num Features: {args.num_features}")
    print(f"  Output Dir: {args.output_dir}")
    
    # Create dataset
    print(f"\n{'='*80}")
    print("CREATING DATASET")
    print(f"{'='*80}")
    X, y = create_dummy_dataset(
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        num_features=args.num_features
    )
    print(f"✅ Created dataset: X.shape={X.shape}, y.shape={y.shape}")
    print(f"Class distribution: {np.bincount(y, minlength=args.num_classes)}")
    
    # Perform cross-validation
    validator = CrossValidator(
        n_splits=args.num_splits,
        num_classes=args.num_classes,
        random_state=args.seed
    )
    
    fold_results = validator.cross_validate(
        X, y,
        model_func=dummy_model_function
    )
    
    # Save results
    output_path = validator.save_results(args.output_dir)
    
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
