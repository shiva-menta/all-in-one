"""
NATTEN compatibility layer for supporting both old (0.17.x) and new (>=0.20) API versions.

The old API has signature: natten1dqkrpb(query, key, rpb, kernel_size, dilation)
The new API has signature: na1d_qk(query, key, kernel_size, dilation, rpb=rpb)

This module provides wrapper functions that accept the new-style keyword rpb argument
and dispatch to the appropriate underlying implementation.
"""

_NATTEN_NEW_API = False

try:
    # Try old API first (natten < 0.20)
    from natten.functional import natten1dqkrpb as _na1d_qk_old
    from natten.functional import natten1dav as _na1d_av_old
    from natten.functional import natten2dqkrpb as _na2d_qk_old
    from natten.functional import natten2dav as _na2d_av_old
    NATTEN_VERSION = "old"
except ImportError:
    # New API (natten >= 0.20)
    from natten.functional import na1d_qk as _na1d_qk_new
    from natten.functional import na1d_av as _na1d_av_new
    from natten.functional import na2d_qk as _na2d_qk_new
    from natten.functional import na2d_av as _na2d_av_new
    _NATTEN_NEW_API = True
    NATTEN_VERSION = "new"


def na1d_qk(query, key, kernel_size, dilation, *, rpb=None):
    """1D neighborhood attention query-key computation with RPB support."""
    if _NATTEN_NEW_API:
        return _na1d_qk_new(query, key, kernel_size, dilation, rpb=rpb)
    else:
        return _na1d_qk_old(query, key, rpb, kernel_size, dilation)


def na1d_av(attn, value, kernel_size, dilation):
    """1D neighborhood attention attention-value computation."""
    if _NATTEN_NEW_API:
        return _na1d_av_new(attn, value, kernel_size, dilation)
    else:
        return _na1d_av_old(attn, value, kernel_size, dilation)


def na2d_qk(query, key, kernel_size, dilation, *, rpb=None):
    """2D neighborhood attention query-key computation with RPB support."""
    if _NATTEN_NEW_API:
        return _na2d_qk_new(query, key, kernel_size, dilation, rpb=rpb)
    else:
        return _na2d_qk_old(query, key, rpb, kernel_size, dilation)


def na2d_av(attn, value, kernel_size, dilation):
    """2D neighborhood attention attention-value computation."""
    if _NATTEN_NEW_API:
        return _na2d_av_new(attn, value, kernel_size, dilation)
    else:
        return _na2d_av_old(attn, value, kernel_size, dilation)


__all__ = ["na1d_qk", "na1d_av", "na2d_qk", "na2d_av", "NATTEN_VERSION"]
