"""
NATTEN 0.20.x compatibility layer for fused neighborhood attention.

NATTEN 0.20+ uses fused operations (na1d, na2d) with heads-last layout.
RPB is no longer supported - models must be retrained.
"""
import torch
from typing import Optional

try:
    from natten import na1d as _na1d
    from natten import na2d as _na2d
    NATTEN_AVAILABLE = True
    NATTEN_VERSION = "0.20+"
except ImportError:
    NATTEN_AVAILABLE = False
    NATTEN_VERSION = None


def check_natten_available():
    if not NATTEN_AVAILABLE:
        raise ImportError(
            "NATTEN >= 0.20.0 is required. "
            "Install with: pip install natten>=0.20.0 "
            "or visit https://natten.org/install/"
        )


def fused_na1d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
) -> torch.Tensor:
    """
    Fused 1D neighborhood attention.
    
    Args:
        query: [B, T, H, D] - heads last layout
        key: [B, T, H, D]
        value: [B, T, H, D]
        kernel_size: int - neighborhood window size
        dilation: int - dilation factor (default: 1)
    
    Returns:
        output: [B, T, H, D] - same layout as input
    """
    check_natten_available()
    return _na1d(query, key, value, kernel_size=kernel_size, dilation=dilation)


def fused_na2d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
) -> torch.Tensor:
    """
    Fused 2D neighborhood attention.
    
    Args:
        query: [B, H, W, heads, D] - heads last layout
        key: [B, H, W, heads, D]
        value: [B, H, W, heads, D]
        kernel_size: int - neighborhood window size
        dilation: int - dilation factor (default: 1)
    
    Returns:
        output: [B, H, W, heads, D] - same layout as input
    """
    check_natten_available()
    return _na2d(query, key, value, kernel_size=kernel_size, dilation=dilation)


__all__ = [
    "fused_na1d",
    "fused_na2d",
    "check_natten_available",
    "NATTEN_AVAILABLE",
    "NATTEN_VERSION",
]
