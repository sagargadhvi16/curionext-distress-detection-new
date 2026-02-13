"""
Audio encoder pruning utilities.

Implements magnitude-based unstructured pruning and reports
effective sparsity instead of raw parameter count.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_sparsity(model: nn.Module) -> float:
    """
    Calculate global sparsity (fraction of zero-valued weights)
    across Linear and Conv layers.
    """
    zero_weights = 0
    total_weights = 0

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            weight = module.weight.data
            zero_weights += torch.sum(weight == 0).item()
            total_weights += weight.numel()

    if total_weights == 0:
        return 0.0

    return zero_weights / total_weights


def apply_magnitude_pruning(
    model: nn.Module,
    amount: float = 0.5,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Apply L1 unstructured pruning to Conv and Linear layers.

    Args:
        model: Audio encoder model
        amount: Fraction of weights to prune (e.g., 0.5 = 50%)
        verbose: Print pruning summary

    Returns:
        Dictionary with pruning statistics
    """
    if not (0.0 < amount < 1.0):
        raise ValueError("Pruning amount must be between 0 and 1")

    total_params = count_parameters(model)

    # Apply magnitude-based pruning
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            prune.l1_unstructured(
                module,
                name="weight",
                amount=amount
            )

    # Remove pruning reparameterization for deployment safety
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if hasattr(module, "weight_orig"):
                prune.remove(module, "weight")

    sparsity = calculate_sparsity(model)

    if verbose:
        print(f"[PRUNING] Total parameters : {total_params}")
        print(f"[PRUNING] Global sparsity  : {sparsity * 100:.2f}%")

    return {
        "total_parameters": total_params,
        "global_sparsity_percent": round(sparsity * 100, 2)
    }


def sanity_check(
    model: nn.Module,
    input_tensor: torch.Tensor
) -> None:
    """
    Run a forward pass to ensure the model remains stable after pruning.
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    if torch.isnan(output).any():
        raise RuntimeError("NaNs detected in model output after pruning")

    print("[PRUNING] Sanity check passed (valid forward pass)")
