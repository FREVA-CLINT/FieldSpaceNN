"""Calendar inputs and rotary positions for monthly climate sequences."""

import numpy as np
import torch


def monthly_time_inputs(timestamps):
    """Return calendar inputs (Unix seconds, month 0..11) and month offsets.

    Inputs must be finite Unix seconds, shaped (T,) or (B,T), with one
    observation per consecutive calendar month. Decode before reducing precision.
    """
    if torch.is_tensor(timestamps):
        timestamps = timestamps.detach().cpu().numpy()
    seconds = np.asarray(timestamps, dtype=np.float64)
    if seconds.ndim == 1:
        seconds = seconds[None, :]
    if seconds.ndim != 2 or not seconds.shape[1] or not np.isfinite(seconds).all():
        raise ValueError("timestamps must be finite Unix seconds with shape (B,T), T > 0")
    # datetime64[s] comfortably covers the climate data's date range.
    months = seconds.astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
    if np.any(np.diff(months, axis=1) != 1):
        raise ValueError("timestamps must represent consecutive calendar months")
    calendar = np.stack((seconds, months % 12), axis=-1)
    positions = months - months[:, :1]
    return torch.tensor(calendar, dtype=torch.float32), torch.tensor(positions, dtype=torch.float32)


def apply_rotary(x, positions, base=10000.0):
    """Rotate packed (B,H,L,D) queries/keys; positions broadcast to (B,1,L)."""
    if x.shape[-1] % 2:
        raise ValueError("Temporal RoPE requires an even head dimension")
    if base <= 0:
        raise ValueError("RoPE base must be positive")
    frequencies = base ** (-torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32) / x.shape[-1])
    angles = positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * frequencies
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
