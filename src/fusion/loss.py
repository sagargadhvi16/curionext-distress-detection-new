"""Multi-task loss function for distress detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining:
    1. Binary Cross-Entropy (BCE) for distress detection
    2. Mean Squared Error (MSE) for severity regression
    3. Cross-Entropy (CE) for distress type classification
    
    Supports weighted combination of losses to balance task importance.
    """
    
    def __init__(
        self,
        distress_weight: float = 1.0,
        severity_weight: float = 0.5,
        type_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize multi-task loss.
        
        Args:
            distress_weight: Weight for binary distress detection loss
            severity_weight: Weight for severity regression loss
            type_weight: Weight for distress type classification loss
            class_weights: Optional tensor of class weights for BCE loss (shape: [2])
                           Higher weight for distress class to reduce false negatives
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        
        self.distress_weight = distress_weight
        self.severity_weight = severity_weight
        self.type_weight = type_weight
        self.reduction = reduction
        
        # Binary Cross-Entropy for distress detection
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            # Default: emphasize distress class (class 1) to reduce false negatives
            self.register_buffer('class_weights', torch.tensor([1.0, 2.0]))
        
        # MSE for severity regression
        self.mse_loss = nn.MSELoss(reduction=reduction)
        
        # Cross-Entropy for distress type classification
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dictionary with keys:
                        - 'distress_logits': (batch_size, 2) - binary classification logits
                        - 'severity': (batch_size, 1) - severity scores (0-10)
                        - 'type_logits': (batch_size, num_types) - type classification logits
            targets: Dictionary with keys:
                     - 'label': (batch_size,) - binary distress labels (0 or 1)
                     - 'severity': (batch_size, 1) - ground truth severity (0-10)
                     - 'distress_type': (batch_size,) - ground truth type labels (0-4)
        
        Returns:
            Dictionary with keys:
            - 'total_loss': Combined weighted loss
            - 'distress_loss': Binary classification loss
            - 'severity_loss': Regression loss
            - 'type_loss': Classification loss
        """
        # Extract predictions
        distress_logits = predictions['distress_logits']
        severity_pred = predictions['severity']
        type_logits = predictions['type_logits']
        
        # Extract targets
        distress_labels = targets['label']
        severity_target = targets['severity']
        type_labels = targets['distress_type']
        
        # Ensure severity has correct shape
        if severity_target.dim() == 1:
            severity_target = severity_target.unsqueeze(-1)
        if severity_pred.dim() == 1:
            severity_pred = severity_pred.unsqueeze(-1)
        
        # 1. Binary Cross-Entropy for distress detection (with class weights)
        distress_loss = F.cross_entropy(
            distress_logits,
            distress_labels,
            weight=self.class_weights,
            reduction=self.reduction
        )
        
        # 2. Mean Squared Error for severity regression
        severity_loss = self.mse_loss(severity_pred, severity_target)
        
        # 3. Cross-Entropy for distress type classification
        type_loss = self.ce_loss(type_logits, type_labels)
        
        # Weighted combination
        total_loss = (
            self.distress_weight * distress_loss +
            self.severity_weight * severity_loss +
            self.type_weight * type_loss
        )
        
        return {
            'total_loss': total_loss,
            'distress_loss': distress_loss,
            'severity_loss': severity_loss,
            'type_loss': type_loss
        }


def create_multitask_loss(config: Optional[Dict] = None) -> MultiTaskLoss:
    """
    Factory function to create MultiTaskLoss from config dictionary.
    
    Args:
        config: Optional configuration dictionary with keys:
                - 'distress_weight': weight for distress loss
                - 'severity_weight': weight for severity loss
                - 'type_weight': weight for type loss
                - 'class_weights': list of class weights [no_distress, distress]
    
    Returns:
        MultiTaskLoss instance
    """
    if config is None:
        return MultiTaskLoss()
    
    class_weights = None
    if 'class_weights' in config:
        class_weights = torch.tensor(config['class_weights'], dtype=torch.float32)
    
    return MultiTaskLoss(
        distress_weight=config.get('distress_weight', 1.0),
        severity_weight=config.get('severity_weight', 0.5),
        type_weight=config.get('type_weight', 0.5),
        class_weights=class_weights
    )

