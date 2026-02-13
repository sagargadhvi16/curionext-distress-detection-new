"""Comprehensive evaluation script for trained model on test dataset."""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fusion.metrics_calculator import MetricsCalculator
from src.utils.visualization import TrainingVisualizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Comprehensive model evaluator."""
    
    # Distress types mapping
    DISTRESS_TYPES = {
        0: "crying",
        1: "fear",
        2: "pain",
        3: "discomfort",
        4: "anger"
    }
    
    def __init__(self, num_classes=5, output_dir="evaluation_results"):
        """Initialize evaluator.
        
        Args:
            num_classes: Number of distress classes
            output_dir: Directory to save evaluation results
        """
        self.num_classes = num_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator(num_classes=num_classes)
        self.visualizer = TrainingVisualizer(output_dir=str(self.output_dir))
        
        self.results = {}
    
    def evaluate(self, y_true, y_pred, y_proba=None):
        """Evaluate model predictions.
        
        Args:
            y_true: Ground truth labels (numpy array)
            y_pred: Predicted labels (numpy array)
            y_proba: Prediction probabilities (numpy array, optional)
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*80}")
        print("OVERALL METRICS")
        print(f"{'='*80}")
        
        # Compute overall metrics
        overall_metrics = self.metrics_calculator.compute_all_metrics(
            y_true, y_pred, y_proba
        )
        
        # Print overall metrics
        print(f"Total Samples: {len(y_true)}")
        print(f"Accuracy: {overall_metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {overall_metrics.get('precision', 0):.4f}")
        print(f"Recall: {overall_metrics.get('recall', 0):.4f}")
        print(f"F1 Score: {overall_metrics.get('f1', 0):.4f}")
        print(f"AUC-ROC: {overall_metrics.get('auc_roc', 0):.4f}")
        print(f"False Negative Rate: {overall_metrics.get('fnr', 0):.4f}")
        
        self.results['overall'] = overall_metrics
        
        # Per-class metrics
        self._evaluate_per_class(y_true, y_pred, y_proba)
        
        # Error analysis
        self._analyze_errors(y_true, y_pred)
        
        # Plot results
        if 'confusion_matrix' in overall_metrics:
            cm = overall_metrics['confusion_matrix']
            class_names = [self.DISTRESS_TYPES.get(i, f"Class {i}") for i in range(self.num_classes)]
            self.visualizer.plot_confusion_matrix(cm, class_names=class_names)
            print(f"\n✅ Confusion matrix saved")
        
        # Plot ROC curve if probabilities provided
        if y_proba is not None and self.num_classes == 2:
            try:
                self.visualizer.plot_roc_curve(y_true, y_proba)
                print(f"✅ ROC curve saved")
            except Exception as e:
                logger.warning(f"Could not plot ROC curve: {e}")
        
        return self.results
    
    def _evaluate_per_class(self, y_true, y_pred, y_proba):
        """Evaluate metrics per distress type."""
        print(f"\n{'='*80}")
        print("PER-CLASS METRICS")
        print(f"{'='*80}\n")
        
        per_class_results = {}
        
        for class_id in range(self.num_classes):
            class_name = self.DISTRESS_TYPES.get(class_id, f"Class {class_id}")
            
            # Binary classification: class_id vs rest
            y_binary_true = (y_true == class_id).astype(int)
            y_binary_pred = (y_pred == class_id).astype(int)
            
            # Get binary probabilities if available
            y_binary_proba = None
            if y_proba is not None:
                if y_proba.ndim == 1:
                    y_binary_proba = y_proba
                else:
                    y_binary_proba = y_proba[:, class_id] if y_proba.shape[1] > class_id else None
            
            metrics = self.metrics_calculator.compute_all_metrics(
                y_binary_true, y_binary_pred, y_binary_proba
            )
            
            per_class_results[class_name] = metrics
            
            # Print per-class metrics
            support = np.sum(y_binary_true)
            print(f"{class_name.upper()} (Class {class_id})")
            print(f"  Support: {int(support)}")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall: {metrics.get('recall', 0):.4f}")
            print(f"  F1 Score: {metrics.get('f1', 0):.4f}")
            print(f"  False Positive Rate: {metrics.get('fpr', 0):.4f}")
            print(f"  False Negative Rate: {metrics.get('fnr', 0):.4f}")
            print()
        
        self.results['per_class'] = per_class_results
    
    def _analyze_errors(self, y_true, y_pred):
        """Analyze false positives and false negatives."""
        print(f"\n{'='*80}")
        print("ERROR ANALYSIS")
        print(f"{'='*80}\n")
        
        # Find errors
        is_error = y_true != y_pred
        correct = y_true == y_pred
        
        total_errors = np.sum(is_error)
        total_correct = np.sum(correct)
        error_rate = total_errors / len(y_true) if len(y_true) > 0 else 0
        
        print(f"Total Correct: {total_correct}")
        print(f"Total Errors: {total_errors}")
        print(f"Error Rate: {error_rate:.4f}\n")
        
        # Analyze false positives and false negatives per class
        error_analysis = {}
        
        for class_id in range(self.num_classes):
            class_name = self.DISTRESS_TYPES.get(class_id, f"Class {class_id}")
            
            # False positives: predicted as class_id but true label is different
            fp_mask = (y_pred == class_id) & (y_true != class_id)
            false_positives = np.sum(fp_mask)
            
            # False negatives: true label is class_id but predicted differently
            fn_mask = (y_true == class_id) & (y_pred != class_id)
            false_negatives = np.sum(fn_mask)
            
            # True positives and negatives
            tp_mask = (y_true == class_id) & (y_pred == class_id)
            true_positives = np.sum(tp_mask)
            
            tn_mask = (y_true != class_id) & (y_pred != class_id)
            true_negatives = np.sum(tn_mask)
            
            error_analysis[class_name] = {
                'true_positives': int(true_positives),
                'true_negatives': int(true_negatives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives)
            }
            
            print(f"{class_name.upper()} (Class {class_id})")
            print(f"  True Positives: {int(true_positives)}")
            print(f"  True Negatives: {int(true_negatives)}")
            print(f"  False Positives: {int(false_positives)}")
            print(f"  False Negatives: {int(false_negatives)}")
            print()
        
        self.results['error_analysis'] = error_analysis
    
    def save_report(self, filename="evaluation_report.json"):
        """Save evaluation report as JSON."""
        report_path = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✅ Evaluation report saved to {report_path}")
    
    def save_csv_report(self, filename="evaluation_metrics.csv"):
        """Save metrics as CSV for easy viewing."""
        csv_path = self.output_dir / filename
        
        # Flatten per-class metrics for CSV
        rows = []
        
        # Overall metrics
        overall = self.results.get('overall', {})
        row = {'Distress Type': 'OVERALL'}
        for metric, value in overall.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                row[metric] = value
        rows.append(row)
        
        # Per-class metrics
        per_class = self.results.get('per_class', {})
        for class_name, metrics in per_class.items():
            row = {'Distress Type': class_name}
            for metric, value in metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    row[metric] = value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        print(f"✅ CSV metrics saved to {csv_path}")


def create_dummy_test_data(num_samples=200, num_classes=5):
    """Create dummy test data for demonstration.
    
    Args:
        num_samples: Number of test samples
        num_classes: Number of distress classes
    
    Returns:
        Tuple of (y_true, y_pred, y_proba)
    """
    # Simulate ground truth
    y_true = np.random.randint(0, num_classes, num_samples)
    
    # Simulate predictions (with 80% accuracy)
    y_pred = y_true.copy()
    error_indices = np.random.choice(
        num_samples,
        size=int(0.2 * num_samples),
        replace=False
    )
    for idx in error_indices:
        wrong_class = np.random.randint(0, num_classes)
        while wrong_class == y_true[idx]:
            wrong_class = np.random.randint(0, num_classes)
        y_pred[idx] = wrong_class
    
    # Simulate probabilities
    y_proba = np.random.dirichlet(np.ones(num_classes), num_samples)
    
    return y_true, y_pred, y_proba


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained model on test dataset")
    parser.add_argument("--num-samples", type=int, default=200,
                       help="Number of test samples (for dummy data)")
    parser.add_argument("--num-classes", type=int, default=5,
                       help="Number of distress classes")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"\n{'='*80}")
    print("MODEL EVALUATION SCRIPT")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Config:")
    print(f"  Num Samples: {args.num_samples}")
    print(f"  Num Classes: {args.num_classes}")
    print(f"  Output Dir: {args.output_dir}")
    
    # Create dummy test data
    print(f"\n{'='*80}")
    print("CREATING DUMMY TEST DATA")
    print(f"{'='*80}")
    y_true, y_pred, y_proba = create_dummy_test_data(
        num_samples=args.num_samples,
        num_classes=args.num_classes
    )
    print(f"✅ Created {len(y_true)} test samples with {args.num_classes} classes")
    
    # Evaluate
    evaluator = Evaluator(
        num_classes=args.num_classes,
        output_dir=args.output_dir
    )
    
    results = evaluator.evaluate(y_true, y_pred, y_proba)
    
    # Save reports
    evaluator.save_report("evaluation_report.json")
    evaluator.save_csv_report("evaluation_metrics.csv")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {evaluator.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
