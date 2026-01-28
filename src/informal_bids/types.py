from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TaskBAuctionObs:
    """Observed auction record used by the Task B estimator (simulated only for now).

    Fields are *observables* only. Derived objects like X_i and [L_i, U_i] are
    produced by the active screening spec during estimation.
    """

    auction_id: int
    bI: np.ndarray  # (J,) observed informal bids
    admitted: np.ndarray  # (J,) bool
    bF: np.ndarray  # (J,) float, NaN if not admitted
    T: np.ndarray  # (J,) int {0:F, 1:S}
    Z: float = 0.0  # exogenous shifter (optional)
    tick: Optional[np.ndarray] = None  # (J,) rounding tick (Candidate 4 proxy)
    tick_alpha_F: float = np.nan  # E[log(tick) | T=0], used to compute precision index
    tick_alpha_S: float = np.nan  # E[log(tick) | T=1], used to compute precision index


@dataclass(frozen=True)
class TaskBDataset:
    """Packed dataset representation (contiguous bidder arrays)."""

    auctions: List[TaskBAuctionObs]
    N: int
    n_bidders: np.ndarray  # (N,)
    offsets: np.ndarray  # (N,)

    # Bidder-level packed arrays (total_bidders,)
    bI: np.ndarray
    bF: np.ndarray
    admitted: np.ndarray
    T: np.ndarray

    # Auction-level arrays (N,)
    Z: np.ndarray
    nS: np.ndarray
    nF: np.ndarray
    g_shareS: np.ndarray
    lambda_f: np.ndarray
    p_prec: np.ndarray  # Candidate 4 precision proxy (type-demeaned, scaled log-tick residual)


def pack_task_b_dataset(auctions: Sequence[TaskBAuctionObs]) -> TaskBDataset:
    auctions = list(auctions)
    N = len(auctions)
    n_bidders = np.array([int(a.bI.size) for a in auctions], dtype=np.int64)

    offsets = np.zeros(N, dtype=np.int64)
    cursor = 0
    for i in range(N):
        offsets[i] = cursor
        cursor += int(n_bidders[i])

    total = int(cursor)
    bI = np.empty(total, dtype=np.float64)
    bF = np.empty(total, dtype=np.float64)
    admitted = np.empty(total, dtype=np.bool_)
    T = np.empty(total, dtype=np.int8)

    Z = np.zeros(N, dtype=np.float64)
    nS = np.zeros(N, dtype=np.int64)
    nF = np.zeros(N, dtype=np.int64)
    g_shareS = np.zeros(N, dtype=np.float64)
    lambda_f = np.zeros(N, dtype=np.float64)
    p_prec = np.full(N, np.nan, dtype=np.float64)

    for i, a in enumerate(auctions):
        n = int(n_bidders[i])
        start = int(offsets[i])
        end = start + n

        bI[start:end] = a.bI.astype(np.float64, copy=False)
        bF[start:end] = a.bF.astype(np.float64, copy=False)
        admitted[start:end] = a.admitted.astype(np.bool_, copy=False)
        T[start:end] = a.T.astype(np.int8, copy=False)

        Z[i] = float(a.Z)
        nS_i = int(np.sum(T[start:end] == 1))
        nS[i] = nS_i
        nF[i] = n - nS_i
        g_shareS[i] = float(nS_i) / float(n) if n > 0 else np.nan
        lambda_f[i] = 1.0 - 1.0 / float(n) if n > 1 else np.nan

        if a.tick is not None:
            tick = np.asarray(a.tick, dtype=float)
            alpha_F = float(getattr(a, "tick_alpha_F", np.nan))
            alpha_S = float(getattr(a, "tick_alpha_S", np.nan))
            if np.isfinite(alpha_F) and np.isfinite(alpha_S) and np.all(tick > 0):
                alpha = np.where(a.T.astype(int) == 1, alpha_S, alpha_F).astype(float)
                p_prec[i] = -10.0 * float(np.mean(np.log(tick) - alpha))
            else:
                # Backward-compatible fallback for older auction objects.
                p_prec[i] = float(np.median(tick))

    return TaskBDataset(
        auctions=auctions,
        N=N,
        n_bidders=n_bidders,
        offsets=offsets,
        bI=bI,
        bF=bF,
        admitted=admitted,
        T=T,
        Z=Z,
        nS=nS,
        nF=nF,
        g_shareS=g_shareS,
        lambda_f=lambda_f,
        p_prec=p_prec,
    )


@dataclass
class TaskBParams:
    beta: np.ndarray
    sigma_omega: float
    gamma: float
    kappa: float
    sigma_nu: float
    sigma_eta: float
    delta: float = 0.0
