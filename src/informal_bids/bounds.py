from __future__ import annotations

from typing import Tuple

import numpy as np


def bounds_baseline(bI: np.ndarray, admitted: np.ndarray) -> Tuple[float, float]:
    """Compute (L,U) from observed admission under baseline rule bI >= b*."""
    admitted = np.asarray(admitted, dtype=bool)
    if not np.any(admitted):
        # Unobserved all-reject; keep a degenerate invalid interval.
        return float(np.inf), float(-np.inf)
    if np.all(admitted):
        return float(-np.inf), float(np.min(bI))
    return float(np.max(bI[~admitted])), float(np.min(bI[admitted]))


def bounds_type_shift(
    bI: np.ndarray,
    admitted: np.ndarray,
    T: np.ndarray,
    *,
    delta: float,
) -> Tuple[float, float]:
    """Compute (L,U) when admission uses bI >= b* + delta*T.

    Using adjusted bids: bI_adj = bI - delta*T.
    """
    admitted = np.asarray(admitted, dtype=bool)
    T = np.asarray(T, dtype=float)
    bI_adj = np.asarray(bI, dtype=float) - float(delta) * T

    if not np.any(admitted):
        return float(np.inf), float(-np.inf)
    if np.all(admitted):
        return float(-np.inf), float(np.min(bI_adj))
    return float(np.max(bI_adj[~admitted])), float(np.min(bI_adj[admitted]))

