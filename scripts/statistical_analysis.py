"""Statistical analysis for model evaluation results."""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from scipy import stats as scipy_stats

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class StatisticalAnalyzer:
    """Statistical analysis of evaluation results."""
    
    def __init__(self, confidence_level=0.95):
        """Initialize analyzer.
        
        Args:
            confidence_level: Confidence level for CI (default 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.results = {}
    
    def analyze_metrics(self, metrics_list, metric_names=None):
        """Analyze metrics across multiple runs/folds.
        
        Args:
            metrics_list: List of metric values or list of dicts
            metric_names: Names of metrics (if metrics_list is list of lists)
        
        Returns:
            Dictionary with statistical analysis
        """
        print(f"\n{'='*80}")
        print("STATISTICAL ANALYSIS")
        print(f"{'='*80}\n")
        
        # Handle different input formats
        if isinstance(metrics_list[0], dict):
            # List of metric dictionaries
            analysis = self._analyze_metric_dicts(metrics_list)
        else:
            # List of metric values
            analysis = self._analyze_metric_values(metrics_list, metric_names)
        
        self.results['metrics_analysis'] = analysis
        return analysis
    
    def _analyze_metric_dicts(self, metric_dicts):
        """Analyze list of metric dictionaries."""
        analysis = {}
        
        # Get all metric names from first dict
        metric_names = list(metric_dicts[0].keys())
        
        for metric_name in metric_names:
            values = []
            for m_dict in metric_dicts:
                if metric_name in m_dict:
                    val = m_dict[metric_name]
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        values.append(float(val))
            
            if values:
                analysis[metric_name] = self._compute_statistics(
                    np.array(values),
                    metric_name
                )
        
        return analysis
    
    def _analyze_metric_values(self, values, metric_names=None):
        """Analyze single metric or multiple metrics."""
        values = np.array(values)
        
        if values.ndim == 1:
            # Single metric
            metric_name = metric_names[0] if metric_names else "metric"
            return {
                metric_name: self._compute_statistics(values, metric_name)
            }
        else:
            # Multiple metrics
            analysis = {}
            for i, metric_values in enumerate(values.T):
                metric_name = metric_names[i] if metric_names and i < len(metric_names) else f"metric_{i}"
                analysis[metric_name] = self._compute_statistics(metric_values, metric_name)
            
            return analysis
    
    def _compute_statistics(self, values, metric_name):
        """Compute comprehensive statistics for a metric.
        
        Args:
            values: Array of metric values
            metric_name: Name of metric for printing
        
        Returns:
            Dictionary with statistics
        """
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample std
        median = np.median(values)
        
        # Confidence interval (95%)
        z_score = scipy_stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = mean - z_score * (std / np.sqrt(n))
        ci_upper = mean + z_score * (std / np.sqrt(n))
        
        # Standard error
        se = std / np.sqrt(n)
        
        stats_dict = {
            'n': int(n),
            'mean': float(mean),
            'std': float(std),
            'se': float(se),
            'median': float(median),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'q1': float(np.percentile(values, 25)),
            'q3': float(np.percentile(values, 75)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ci_level': self.confidence_level
        }
        
        # Print statistics
        print(f"{metric_name.upper()}")
        print(f"  N: {int(n)}")
        print(f"  Mean: {mean:.4f}")
        print(f"  Std Dev: {std:.4f}")
        print(f"  Median: {median:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Min: {np.min(values):.4f}, Max: {np.max(values):.4f}")
        print()
        
        return stats_dict
    
    def perform_t_test(self, group1, group2, metric_name="comparison"):
        """Perform independent samples t-test.
        
        Args:
            group1: First group of values
            group2: Second group of values
            metric_name: Name for printing
        
        Returns:
            Dictionary with t-test results
        """
        print(f"\n{'='*80}")
        print("INDEPENDENT SAMPLES T-TEST")
        print(f"{'='*80}\n")
        
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        # Perform t-test
        t_stat, p_value = scipy_stats.ttest_ind(group1, group2)
        
        # Cohen's d (effect size)
        cohens_d = (np.mean(group1) - np.mean(group2)) / np.sqrt(
            ((len(group1)-1)*np.std(group1, ddof=1)**2 + 
             (len(group2)-1)*np.std(group2, ddof=1)**2) / 
            (len(group1) + len(group2) - 2)
        )
        
        # Welch's t-test (doesn't assume equal variances)
        t_stat_welch, p_value_welch = scipy_stats.ttest_ind(group1, group2, equal_var=False)
        
        result = {
            'test_type': 'independent_samples_ttest',
            'metric': metric_name,
            'group1_size': len(group1),
            'group2_size': len(group2),
            'group1_mean': float(np.mean(group1)),
            'group2_mean': float(np.mean(group2)),
            'group1_std': float(np.std(group1, ddof=1)),
            'group2_std': float(np.std(group2, ddof=1)),
            'students_t_statistic': float(t_stat),
            'students_p_value': float(p_value),
            'welch_t_statistic': float(t_stat_welch),
            'welch_p_value': float(p_value_welch),
            'cohens_d': float(cohens_d),
            'significant_at_0.05': float(p_value) < 0.05,
            'significant_at_0.01': float(p_value) < 0.01
        }
        
        print(f"{metric_name.upper()} - Independent Samples T-Test")
        print(f"  Group 1: n={len(group1)}, mean={np.mean(group1):.4f}, std={np.std(group1, ddof=1):.4f}")
        print(f"  Group 2: n={len(group2)}, mean={np.mean(group2):.4f}, std={np.std(group2, ddof=1):.4f}")
        print(f"  Student's T: {t_stat:.4f}, p-value: {p_value:.6f}")
        print(f"  Welch's T: {t_stat_welch:.4f}, p-value: {p_value_welch:.6f}")
        print(f"  Cohen's d: {cohens_d:.4f}")
        print(f"  Significant at 0.05: {'Yes' if p_value < 0.05 else 'No'}")
        print()
        
        self.results['t_test'] = result
        return result
    
    def perform_paired_t_test(self, before, after, metric_name="comparison"):
        """Perform paired samples t-test.
        
        Args:
            before: Before values
            after: After values
            metric_name: Name for printing
        
        Returns:
            Dictionary with paired t-test results
        """
        print(f"\n{'='*80}")
        print("PAIRED SAMPLES T-TEST")
        print(f"{'='*80}\n")
        
        before = np.array(before)
        after = np.array(after)
        
        if len(before) != len(after):
            raise ValueError("Before and after arrays must have same length")
        
        differences = after - before
        
        # Perform paired t-test
        t_stat, p_value = scipy_stats.ttest_rel(before, after)
        
        # Effect size
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)
        
        result = {
            'test_type': 'paired_ttest',
            'metric': metric_name,
            'n_pairs': len(before),
            'before_mean': float(np.mean(before)),
            'after_mean': float(np.mean(after)),
            'before_std': float(np.std(before, ddof=1)),
            'after_std': float(np.std(after, ddof=1)),
            'mean_difference': float(np.mean(differences)),
            'std_difference': float(np.std(differences, ddof=1)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant_at_0.05': float(p_value) < 0.05,
            'significant_at_0.01': float(p_value) < 0.01
        }
        
        print(f"{metric_name.upper()} - Paired Samples T-Test")
        print(f"  N pairs: {len(before)}")
        print(f"  Before: mean={np.mean(before):.4f}, std={np.std(before, ddof=1):.4f}")
        print(f"  After: mean={np.mean(after):.4f}, std={np.std(after, ddof=1):.4f}")
        print(f"  Mean difference: {np.mean(differences):.4f} ± {np.std(differences, ddof=1):.4f}")
        print(f"  T: {t_stat:.4f}, p-value: {p_value:.6f}")
        print(f"  Cohen's d: {cohens_d:.4f}")
        print(f"  Significant at 0.05: {'Yes' if p_value < 0.05 else 'No'}")
        print()
        
        self.results['paired_t_test'] = result
        return result
    
    def perform_anova(self, *groups, metric_name="comparison"):
        """Perform one-way ANOVA.
        
        Args:
            *groups: Variable number of groups
            metric_name: Name for printing
        
        Returns:
            Dictionary with ANOVA results
        """
        print(f"\n{'='*80}")
        print("ONE-WAY ANOVA")
        print(f"{'='*80}\n")
        
        groups = [np.array(g) for g in groups]
        
        # Perform ANOVA
        f_stat, p_value = scipy_stats.f_oneway(*groups)
        
        result = {
            'test_type': 'anova',
            'metric': metric_name,
            'num_groups': len(groups),
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant_at_0.05': float(p_value) < 0.05,
            'significant_at_0.01': float(p_value) < 0.01,
            'group_stats': []
        }
        
        print(f"{metric_name.upper()} - One-Way ANOVA")
        print(f"  Number of groups: {len(groups)}")
        
        for i, group in enumerate(groups):
            group_stats = {
                'group': i + 1,
                'n': len(group),
                'mean': float(np.mean(group)),
                'std': float(np.std(group, ddof=1))
            }
            result['group_stats'].append(group_stats)
            print(f"  Group {i+1}: n={len(group)}, mean={np.mean(group):.4f}, std={np.std(group, ddof=1):.4f}")
        
        print(f"  F: {f_stat:.4f}, p-value: {p_value:.6f}")
        print(f"  Significant at 0.05: {'Yes' if p_value < 0.05 else 'No'}")
        print()
        
        self.results['anova'] = result
        return result
    
    def save_results(self, output_dir="statistical_analysis"):
        """Save analysis results as JSON.
        
        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['confidence_level'] = self.confidence_level
        
        # Save to JSON
        json_path = output_path / "statistical_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Statistical analysis saved to {json_path}")
        
        return output_path


def create_example_datasets(num_samples=100, num_folds=5):
    """Create example datasets for statistical testing.
    
    Args:
        num_samples: Number of samples per fold
        num_folds: Number of folds
    
    Returns:
        Dictionary with example datasets
    """
    np.random.seed(42)
    
    # Simulated metrics from cross-validation folds
    accuracy_per_fold = np.random.beta(9, 1, num_folds)  # High accuracy
    f1_per_fold = np.random.beta(8.5, 1, num_folds)
    fnr_per_fold = np.random.beta(2, 8, num_folds)  # Low FNR
    
    # Simulated comparison: baseline vs improved model
    baseline_accuracy = np.random.normal(0.80, 0.05, num_folds)
    improved_accuracy = np.random.normal(0.88, 0.04, num_folds)
    
    return {
        'accuracy_per_fold': accuracy_per_fold,
        'f1_per_fold': f1_per_fold,
        'fnr_per_fold': fnr_per_fold,
        'baseline_accuracy': baseline_accuracy,
        'improved_accuracy': improved_accuracy
    }


def main():
    """Main statistical analysis function."""
    parser = argparse.ArgumentParser(description="Statistical analysis of evaluation results")
    parser.add_argument("--confidence-level", type=float, default=0.95,
                       help="Confidence level (default 0.95)")
    parser.add_argument("--output-dir", type=str, default="statistical_analysis",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS SCRIPT")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Config:")
    print(f"  Confidence Level: {args.confidence_level}")
    print(f"  Output Dir: {args.output_dir}")
    
    # Create example datasets
    print(f"\n{'='*80}")
    print("CREATING EXAMPLE DATASETS")
    print(f"{'='*80}")
    datasets = create_example_datasets()
    print(f"✅ Created example datasets from 5-fold cross-validation")
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(confidence_level=args.confidence_level)
    
    # Analyze individual metrics
    print(f"\n{'='*80}")
    print("ANALYZING INDIVIDUAL METRICS")
    print(f"{'='*80}")
    
    analyzer.analyze_metrics(
        datasets['accuracy_per_fold'],
        metric_names=['Accuracy']
    )
    analyzer.analyze_metrics(
        datasets['f1_per_fold'],
        metric_names=['F1 Score']
    )
    analyzer.analyze_metrics(
        datasets['fnr_per_fold'],
        metric_names=['False Negative Rate']
    )
    
    # Perform t-tests
    analyzer.perform_t_test(
        datasets['baseline_accuracy'],
        datasets['improved_accuracy'],
        metric_name='Accuracy (Baseline vs Improved)'
    )
    
    # Perform paired t-test
    analyzer.perform_paired_t_test(
        datasets['baseline_accuracy'],
        datasets['improved_accuracy'],
        metric_name='Accuracy (Baseline vs Improved)'
    )
    
    # Save results
    analyzer.save_results(args.output_dir)
    
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
