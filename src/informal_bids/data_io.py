"""
Data I/O for real auction data.

This module provides utilities for loading real auction data from files,
enabling the MCMC samplers to work with empirical data in addition to
simulated data.

The samplers are data-source agnostic - they work with AuctionData objects
regardless of whether they came from simulation (DataGenerator) or real
files (RealDataLoader).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

from .data import TaskAAuctionData, TaskBAuctionData


class RealDataLoader:
    """Load real auction data from CSV files.

    This class provides methods to load auction data with bounds [L_i, U_i]
    and covariates X_i from CSV files, producing the same AuctionData objects
    that the DataGenerator classes produce.

    Example CSV format for Task A:
        auction_id,lower_bound,upper_bound,covariate_1,covariate_2
        0,1.2,1.5,0.3,-0.2
        1,1.1,1.4,0.1,0.5

    Task B note (two-stage DGP):
        The current Task B estimator uses bidder-level informal bids and admitted-only
        formal bids. A general-purpose CSV loader for that long-format structure is
        not implemented yet.
    """

    @staticmethod
    def load_task_a_data(
        filepath: str,
        covariate_cols: Optional[List[str]] = None,
        lower_bound_col: str = 'lower_bound',
        upper_bound_col: str = 'upper_bound',
        auction_id_col: str = 'auction_id'
    ) -> Tuple[List[TaskAAuctionData], dict]:
        """Load Task A auction data from CSV.

        Args:
            filepath: Path to CSV file
            covariate_cols: List of column names to use as covariates.
                           If None, uses intercept only (k=1).
            lower_bound_col: Column name for lower bound
            upper_bound_col: Column name for upper bound
            auction_id_col: Column name for auction ID

        Returns:
            Tuple of (list of TaskAAuctionData, summary dict)
        """
        df = pd.read_csv(filepath)

        auctions = []
        n_complete = 0
        n_incomplete = 0

        for idx, row in df.iterrows():
            # Build covariate vector with intercept
            if covariate_cols:
                covs = [row[c] for c in covariate_cols]
                X_i = np.array([1.0] + covs)
            else:
                X_i = np.array([1.0])

            L_i = row[lower_bound_col]
            U_i = row[upper_bound_col]

            # Handle missing/infinite bounds
            if pd.isna(L_i):
                L_i = -np.inf
            if pd.isna(U_i):
                U_i = np.inf

            is_complete = np.isfinite(L_i) and np.isfinite(U_i)
            if is_complete:
                n_complete += 1
            else:
                n_incomplete += 1

            auction = TaskAAuctionData(
                auction_id=int(row[auction_id_col]) if auction_id_col in df.columns else idx,
                X_i=X_i,
                bids=np.array([]),  # Real data may not have individual bids
                admitted=np.array([]),
                L_i=float(L_i),
                U_i=float(U_i),
                is_complete=is_complete
            )
            auctions.append(auction)

        summary = {
            'n_total': len(auctions),
            'n_complete': n_complete,
            'n_incomplete': n_incomplete,
            'pct_incomplete': 100 * n_incomplete / len(auctions) if auctions else 0,
            'k_covariates': auctions[0].X_i.shape[0] if auctions else 1
        }

        return auctions, summary

    @staticmethod
    def load_task_b_data(
        filepath: str,
        **_kwargs,
    ) -> Tuple[List[TaskBAuctionData], dict]:
        """Load Task B auction data from CSV (not implemented).

        Task B (two-stage) requires bidder-level informal bids and admitted-only
        formal bids. Until a long-format loader is defined, use the simulation
        generators in `data.py`.
        """
        raise NotImplementedError(
            "Task B two-stage CSV loader is not implemented. "
            "Use `TaskBDataGenerator` or implement a long-format loader for bidder-level bids."
        )

    @staticmethod
    def validate_bounds(auctions: List, task: str = 'A') -> bool:
        """Validate that bounds are consistent (L <= U).

        Args:
            auctions: List of auction data objects
            task: 'A' or 'B' to specify which task format

        Returns:
            True if all bounds are valid, raises ValueError otherwise
        """
        for a in auctions:
            if task == 'A':
                if np.isfinite(a.L_i) and np.isfinite(a.U_i) and a.L_i > a.U_i:
                    raise ValueError(
                        f"Auction {a.auction_id}: Invalid bounds L={a.L_i} > U={a.U_i}")
            elif task == 'B':
                if np.isfinite(a.L_i) and np.isfinite(a.U_i) and a.L_i > a.U_i:
                    raise ValueError(
                        f"Auction {a.auction_id}: Invalid bounds L={a.L_i} > U={a.U_i}")
        return True
