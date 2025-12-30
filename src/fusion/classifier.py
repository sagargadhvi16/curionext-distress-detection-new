"""Multi-task classification head for distress detection."""
import torch
import torch.nn as nn


class MultiTaskClassifier(nn.Module):
    """
    Multi-task classifier for distress detection with 3 output heads:
    1. Binary distress detection (distress/no_distress)
    2. Severity regression (0-10 scale)
    3. Distress type classification (5 classes: crying, fear, pain, verbal_abuse, emergency)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        num_distress_types: int = 5
    ):
        """
        Initialize multi-task classifier.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            num_distress_types: Number of distress type classes (default: 5)
        """
        super().__init__()
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task 1: Binary distress detection (2 classes)
        self.distress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # binary classification
        )
        
        # Task 2: Severity regression (0-10 scale)
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # regression output
            nn.Sigmoid()  # Scale to 0-1, then multiply by 10
        )
        
        # Task 3: Distress type classification (5 classes)
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_distress_types)
        )
    
    def forward(self, embeddings: torch.Tensor) -> dict:
        """
        Forward pass with multi-task outputs.

        Args:
            embeddings: Fused embeddings (batch_size, input_dim)

        Returns:
            Dictionary with keys:
            - distress_logits: (batch_size, 2) - binary classification logits
            - severity: (batch_size, 1) - severity score (0-1, multiply by 10 for actual score)
            - type_logits: (batch_size, num_distress_types) - distress type classification logits
        """
        # Shared feature extraction
        shared_features = self.shared_layers(embeddings)
        
        # Task-specific heads
        distress_logits = self.distress_head(shared_features)
        severity = self.severity_head(shared_features)  # 0-1 range
        type_logits = self.type_head(shared_features)
        
        return {
            'distress_logits': distress_logits,
            'severity': severity * 10.0,  # Scale to 0-10
            'type_logits': type_logits
        }


# Keep old class name for backward compatibility
DistressClassifier = MultiTaskClassifier
