"""Temporal Convolutional Network (TCN) for audio feature refinement - adapted from WER-SSL."""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import Optional


class Chomp1d(nn.Module):
    """Remove padding from right side of temporal convolution output."""
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, seq_len)
        Returns:
            x without last chomp_size timesteps
        """
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Temporal convolutional block with residual connection.
    
    Adapted from: https://github.com/locuslab/TCN
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        
        # First convolution with weight normalization
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution with weight normalization
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection: 1x1 conv if dimensions mismatch
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        self.init_weights()

    def init_weights(self):
        """Initialize weights with small normal distribution."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_inputs, seq_len)
        Returns:
            (batch_size, n_outputs, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Multi-layer Temporal Convolutional Network.
    
    Uses dilated convolutions to capture long-range temporal dependencies.
    Adapted from: https://github.com/locuslab/TCN
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: list,
        kernel_size: int = 5,
        dropout: float = 0.2
    ):
        """
        Initialize TCN.
        
        Args:
            num_inputs: Number of input channels
            num_channels: List of output channels for each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_inputs, seq_len)
        Returns:
            (batch_size, num_channels[-1], seq_len)
        """
        return self.network(x)


class TCNAudioRefiner(nn.Module):
    """
    Audio refinement module using Temporal Convolutional Networks.
    
    Takes YAMNet embeddings and refines them to better capture
    temporal patterns specific to distress detection.
    
    Architecture:
    1. TCN layers with increasing receptive fields
    2. Project to refined embedding dimension
    3. Optional temporal pooling for aggregate embedding
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        tcn_channels: list = [256, 256, 128],
        kernel_size: int = 5,
        dropout: float = 0.2,
        output_dim: int = 256,
        use_temporal_pooling: bool = False
    ):
        """
        Initialize TCN audio refiner.
        
        Args:
            input_dim: Input embedding dimension (YAMNet = 1024)
            tcn_channels: List of channel dimensions for TCN layers
            kernel_size: Kernel size for TCN convolutions
            dropout: Dropout probability
            output_dim: Output embedding dimension
            use_temporal_pooling: Whether to pool temporal dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_temporal_pooling = use_temporal_pooling
        
        # Project input to first TCN channel dimension
        self.input_project = nn.Linear(input_dim, tcn_channels[0])
        
        # TCN layers
        self.tcn = TemporalConvNet(
            num_inputs=tcn_channels[0],
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Output projection
        self.output_project = nn.Linear(tcn_channels[-1], output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refine audio embeddings through TCN.
        
        Args:
            x: Audio embeddings
                - If (batch_size, seq_len, input_dim): temporal embeddings
                - If (batch_size, input_dim): aggregate embedding (will add seq_len=1)
        
        Returns:
            Refined embeddings with same temporal structure as input
            - If temporal input: (batch_size, seq_len, output_dim)
            - If aggregate input: (batch_size, output_dim)
        """
        is_temporal = x.dim() == 3
        
        if not is_temporal:
            # Aggregate embedding: add time dimension
            x = x.unsqueeze(1)  # (B, 1, input_dim)
        
        batch_size, seq_len, _ = x.shape
        
        # Project to TCN input dimension: (B, seq_len, tcn_channels[0])
        x = self.input_project(x)
        
        # Rearrange for TCN: (B, tcn_channels[0], seq_len)
        x = x.transpose(1, 2)
        
        # Apply TCN
        x = self.tcn(x)  # (B, tcn_channels[-1], seq_len)
        
        # Rearrange back: (B, seq_len, tcn_channels[-1])
        x = x.transpose(1, 2)
        
        # Project to output dimension: (B, seq_len, output_dim)
        x = self.output_project(x)
        x = self.output_norm(x)
        
        if self.use_temporal_pooling:
            # Average over temporal dimension
            x = x.mean(dim=1)  # (B, output_dim)
        elif not is_temporal:
            # Remove added time dimension if input was aggregate
            x = x.squeeze(1)  # (B, output_dim)
        
        return x

    def extract_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract refined temporal features without pooling.
        
        Returns temporal features for use in attention-based fusion.
        """
        original_use_pooling = self.use_temporal_pooling
        self.use_temporal_pooling = False
        
        output = self.forward(x)
        
        self.use_temporal_pooling = original_use_pooling
        return output
