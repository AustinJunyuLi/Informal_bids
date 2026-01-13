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

from .data import TaskAAuctionData, TaskBAuctionData, BidderData


class RealDataLoader:
    """Load real auction data from CSV files.

    This class provides methods to load auction data with bounds [L_i, U_i]
    and covariates X_i from CSV files, producing the same AuctionData objects
    that the DataGenerator classes produce.

    Example CSV format for Task A:
        auction_id,lower_bound,upper_bound,covariate_1,covariate_2
        0,1.2,1.5,0.3,-0.2
        1,1.1,1.4,0.1,0.5

    Example CSV format for Task B:
        auction_id,L_S,U_S,L_F,U_F,covariate_1,covariate_2
        0,1.3,1.6,1.1,1.4,0.3,-0.2
        1,1.2,1.5,1.0,1.3,0.1,0.5
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
        covariate_cols: Optional[List[str]] = None,
        L_S_col: str = 'L_S',
        U_S_col: str = 'U_S',
        L_F_col: str = 'L_F',
        U_F_col: str = 'U_F',
        auction_id_col: str = 'auction_id'
    ) -> Tuple[List[TaskBAuctionData], dict]:
        """Load Task B auction data from CSV.

        Args:
            filepath: Path to CSV file
            covariate_cols: List of column names to use as covariates
            L_S_col, U_S_col: Column names for Type S bounds
            L_F_col, U_F_col: Column names for Type F bounds
            auction_id_col: Column name for auction ID

        Returns:
            Tuple of (list of TaskBAuctionData, summary dict)
        """
        df = pd.read_csv(filepath)

        auctions = []
        n_complete_S = 0
        n_complete_F = 0
        n_complete_both = 0

        for idx, row in df.iterrows():
            # Build covariate vector with intercept
            if covariate_cols:
                covs = [row[c] for c in covariate_cols]
                X_i = np.array([1.0] + covs)
            else:
                X_i = np.array([1.0])

            # Type S bounds
            L_S = row[L_S_col] if L_S_col in df.columns else -np.inf
            U_S = row[U_S_col] if U_S_col in df.columns else np.inf
            if pd.isna(L_S):
                L_S = -np.inf
            if pd.isna(U_S):
                U_S = np.inf
            has_S_bounds = np.isfinite(L_S) and np.isfinite(U_S)

            # Type F bounds
            L_F = row[L_F_col] if L_F_col in df.columns else -np.inf
            U_F = row[U_F_col] if U_F_col in df.columns else np.inf
            if pd.isna(L_F):
                L_F = -np.inf
            if pd.isna(U_F):
                U_F = np.inf
            has_F_bounds = np.isfinite(L_F) and np.isfinite(U_F)

            if has_S_bounds:
                n_complete_S += 1
            if has_F_bounds:
                n_complete_F += 1
            if has_S_bounds and has_F_bounds:
                n_complete_both += 1

            auction = TaskBAuctionData(
                auction_id=int(row[auction_id_col]) if auction_id_col in df.columns else idx,
                X_i=X_i,
                bidders=[],  # Real data may not have individual bidders
                L_S=float(L_S),
                U_S=float(U_S),
                L_F=float(L_F),
                U_F=float(U_F),
                has_S_bounds=has_S_bounds,
                has_F_bounds=has_F_bounds
            )
            auctions.append(auction)

        summary = {
            'n_total': len(auctions),
            'n_complete_both': n_complete_both,
            'n_complete_S_only': n_complete_S,
            'n_complete_F_only': n_complete_F,
            'k_covariates': auctions[0].X_i.shape[0] if auctions else 1
        }

        return auctions, summary

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
                if a.has_S_bounds and a.L_S > a.U_S:
                    raise ValueError(
                        f"Auction {a.auction_id}: Invalid S bounds L_S={a.L_S} > U_S={a.U_S}")
                if a.has_F_bounds and a.L_F > a.U_F:
                    raise ValueError(
                        f"Auction {a.auction_id}: Invalid F bounds L_F={a.L_F} > U_F={a.U_F}")
        return True
