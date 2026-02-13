"""Explainability module for fusion layer including attention visualization and confidence calculation."""
import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization features will be limited.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Some explainability features will be limited.")


def visualize_attention_weights(
    attention_weights: Union[torch.Tensor, np.ndarray],
    modalities: Optional[list] = None,
    sample_indices: Optional[list] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "YlOrRd"
):
    """
    Visualize cross-modal attention weights with heatmaps.
    
    Shows which modality (audio, biometric, context) contributed most to the prediction.
    
    Args:
        attention_weights: Attention weights tensor/array of shape (batch_size, 3) 
                          or (3,) for single sample. Weights should sum to 1 per sample.
        modalities: List of modality names (default: ['Audio', 'Biometric', 'Context'])
        sample_indices: Optional list of sample indices/names for x-axis labels
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        cmap: Colormap for heatmap
        
    Returns:
        Matplotlib figure object (if matplotlib is available)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualization. Install with: pip install matplotlib seaborn")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert to numpy if torch tensor
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Handle single sample (1D -> 2D)
    if attention_weights.ndim == 1:
        attention_weights = attention_weights.reshape(1, -1)
    
    batch_size, num_modalities = attention_weights.shape
    
    if num_modalities != 3:
        raise ValueError(f"Expected 3 modalities, got {num_modalities}")
    
    # Default modality names
    if modalities is None:
        modalities = ['Audio', 'Biometric', 'Context']
    
    if len(modalities) != num_modalities:
        raise ValueError(f"Number of modality names ({len(modalities)}) must match number of modalities ({num_modalities})")
    
    # Default sample labels
    if sample_indices is None:
        sample_indices = [f'Sample {i+1}' for i in range(batch_size)]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Heatmap of attention weights
    sns.heatmap(
        attention_weights.T,  # Transpose: modalities as rows, samples as columns
        annot=True,
        fmt='.3f',
        cmap=cmap,
        xticklabels=sample_indices,
        yticklabels=modalities,
        ax=ax1,
        cbar_kws={'label': 'Attention Weight'}
    )
    ax1.set_title('Cross-Modal Attention Weights', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Samples', fontsize=12)
    ax1.set_ylabel('Modalities', fontsize=12)
    
    # 2. Bar plot showing average attention weights
    avg_weights = attention_weights.mean(axis=0)
    bars = ax2.bar(modalities, avg_weights, color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
    ax2.set_title('Average Attention Weights', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Weight', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, weight in zip(bars, avg_weights):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{weight:.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    
    return fig


class DistressExplainer:
    """
    SHAP-based explainer for distress detection model predictions.
    """
    
    def __init__(self, model: torch.nn.Module, background_data: Optional[torch.Tensor] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained distress detection model
            background_data: Background dataset for SHAP (optional)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        self.model = model
        self.background_data = background_data
        self.explainer = None
        
        if background_data is not None:
            self._init_explainer()
    
    def _init_explainer(self):
        """Initialize SHAP explainer with background data."""
        # Wrap model for SHAP
        def model_wrapper(x):
            """Wrapper function for SHAP."""
            self.model.eval()
            with torch.no_grad():
                # Assume x is the fused embedding
                output = self.model.classifier(x)
                # Return distress probability
                probs = torch.softmax(output['distress_logits'], dim=-1)
                return probs[:, 1].cpu().numpy()  # Return probability of distress class
        
        self.explainer = shap.Explainer(
            model_wrapper,
            self.background_data
        )
    
    def explain_prediction(
        self,
        input_data: torch.Tensor,
        show_plot: bool = True
    ) -> Dict:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            input_data: Input fused embedding (1, embedding_dim) or (embedding_dim,)
            show_plot: Whether to display SHAP plot
            
        Returns:
            Dictionary with SHAP values and explanations
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Provide background_data during initialization.")
        
        # Ensure correct shape
        if input_data.ndim == 1:
            input_data = input_data.unsqueeze(0)
        
        # Convert to numpy if needed
        if torch.is_tensor(input_data):
            input_data = input_data.detach().cpu().numpy()
        
        # Compute SHAP values
        shap_values = self.explainer(input_data)
        
        if show_plot:
            shap.plots.waterfall(shap_values[0])
        
        return {
            'shap_values': shap_values,
            'base_value': shap_values.base_values,
            'data': shap_values.data
        }
    
    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[list] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values array
            feature_names: Optional list of feature names
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed.")
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for visualization.")
        
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def calculate_confidence(
    model_outputs: Dict[str, torch.Tensor],
    attention_weights: Optional[torch.Tensor] = None,
    rule_alerts: Optional[list] = None,
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    Calculate prediction confidence scores based on model uncertainty and rule agreement.
    
    Uses multiple factors:
    1. Model output probability/entropy (uncertainty)
    2. Agreement between modalities (attention weight distribution)
    3. Rule-based system agreement
    
    Args:
        model_outputs: Dictionary with model outputs containing:
                      - 'distress_logits': (batch_size, 2)
                      - 'severity': (batch_size, 1)
                      - 'type_logits': (batch_size, num_types)
        attention_weights: Optional attention weights (batch_size, 3) for modality agreement
        rule_alerts: Optional list of Alert objects from RuleEngine
        temperature: Temperature scaling for uncertainty (higher = more uncertain)
        
    Returns:
        Dictionary with confidence scores:
        - 'distress_confidence': Confidence in distress prediction (0-1)
        - 'severity_confidence': Confidence in severity prediction (0-1)
        - 'type_confidence': Confidence in distress type prediction (0-1)
        - 'overall_confidence': Overall confidence (0-1)
        - 'modality_agreement': Agreement between modalities (0-1)
        - 'rule_agreement': Agreement with rule-based system (0-1)
    """
    # Convert tensors to numpy if needed
    distress_logits = model_outputs['distress_logits']
    if torch.is_tensor(distress_logits):
        distress_logits = distress_logits.detach().cpu().numpy()
    
    severity = model_outputs['severity']
    if torch.is_tensor(severity):
        severity = severity.detach().cpu().numpy()
    
    type_logits = model_outputs['type_logits']
    if torch.is_tensor(type_logits):
        type_logits = type_logits.detach().cpu().numpy()
    
    # Handle batch dimension
    if distress_logits.ndim == 2 and distress_logits.shape[0] == 1:
        distress_logits = distress_logits[0]
        severity = severity[0] if severity.ndim > 1 else severity
        type_logits = type_logits[0]
    
    # 1. Calculate uncertainty from model outputs using entropy
    # Distress prediction confidence (based on softmax entropy)
    distress_probs = torch.softmax(torch.tensor(distress_logits) / temperature, dim=-1).numpy()
    distress_entropy = -np.sum(distress_probs * np.log(distress_probs + 1e-10))
    max_entropy = np.log(2)  # Binary classification
    distress_confidence = 1.0 - (distress_entropy / max_entropy)
    distress_confidence = np.clip(distress_confidence, 0.0, 1.0)
    
    # Severity confidence (based on output variance - for regression, use fixed confidence or input variance)
    # For now, use a fixed moderate confidence for severity
    severity_confidence = 0.7  # Can be improved with actual variance estimation
    
    # Type prediction confidence
    type_probs = torch.softmax(torch.tensor(type_logits) / temperature, dim=-1).numpy()
    type_entropy = -np.sum(type_probs * np.log(type_probs + 1e-10))
    max_type_entropy = np.log(len(type_probs))  # Multi-class entropy
    type_confidence = 1.0 - (type_entropy / max_type_entropy)
    type_confidence = np.clip(type_confidence, 0.0, 1.0)
    
    # 2. Modality agreement (based on attention weights)
    modality_agreement = 1.0
    if attention_weights is not None:
        if torch.is_tensor(attention_weights):
            attn = attention_weights.detach().cpu().numpy()
        else:
            attn = attention_weights
        
        # Handle batch dimension
        if attn.ndim == 2 and attn.shape[0] == 1:
            attn = attn[0]
        
        # Higher agreement if weights are balanced (lower variance in weights)
        # Lower agreement if one modality dominates (high variance)
        weight_variance = np.var(attn)
        max_variance = 0.25  # Maximum variance for 3 weights summing to 1
        modality_agreement = 1.0 - (weight_variance / max_variance)
        modality_agreement = np.clip(modality_agreement, 0.0, 1.0)
    
    # 3. Rule agreement
    rule_agreement = 1.0
    if rule_alerts is not None:
        # If no alerts, high agreement (normal state)
        if len(rule_alerts) == 0:
            rule_agreement = 1.0
        else:
            # Check if model prediction aligns with rule alerts
            # If model predicts distress and rules detect issues -> high agreement
            # If model predicts no distress but rules detect issues -> low agreement
            predicted_distress = np.argmax(distress_logits) == 1
            has_critical_alerts = any(alert.level.value in ['critical', 'emergency'] for alert in rule_alerts)
            
            if predicted_distress and has_critical_alerts:
                rule_agreement = 0.9  # High agreement
            elif not predicted_distress and not has_critical_alerts:
                rule_agreement = 0.9  # High agreement (both say normal)
            else:
                rule_agreement = 0.5  # Low agreement (disagreement)
    
    # Overall confidence (weighted average)
    overall_confidence = (
        0.4 * distress_confidence +
        0.2 * severity_confidence +
        0.2 * type_confidence +
        0.1 * modality_agreement +
        0.1 * rule_agreement
    )
    overall_confidence = np.clip(overall_confidence, 0.0, 1.0)
    
    return {
        'distress_confidence': float(distress_confidence),
        'severity_confidence': float(severity_confidence),
        'type_confidence': float(type_confidence),
        'overall_confidence': float(overall_confidence),
        'modality_agreement': float(modality_agreement),
        'rule_agreement': float(rule_agreement)
    }

