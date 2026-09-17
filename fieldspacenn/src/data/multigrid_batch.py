"""Compatibility helpers for multigrid dataset batches."""

from __future__ import annotations

from typing import Any, Optional, Tuple


def unpack_multigrid_batch(batch: Tuple[Any, ...]) -> Tuple[Any, Any, Any, Any, Any, Optional[Any]]:
    """Normalize legacy five-field and regular six-field batches."""
    if len(batch) == 5:
        source, target, attention_masks, embeddings, patch_indices = batch
        return source, target, attention_masks, embeddings, patch_indices, None
    if len(batch) == 6:
        return batch
    raise ValueError(f"Expected a five- or six-field multigrid batch, got {len(batch)} fields.")
