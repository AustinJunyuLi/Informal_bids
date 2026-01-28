from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


MisreportingMode = Literal["scale", "shift"]


def lambda_f_first_price(n_bidders: int) -> float:
    if n_bidders <= 1:
        raise ValueError("n_bidders must be >= 2")
    return float(1.0 - 1.0 / float(n_bidders))


def informal_bid_multiplier(
    n_bidders: int,
    kappa: float,
    *,
    mode: MisreportingMode = "scale",
) -> float:
    if mode == "scale":
        return float(lambda_f_first_price(n_bidders) * np.exp(float(kappa)))
    if mode == "shift":
        return float(np.exp(float(kappa)))
    raise ValueError("mode must be 'scale' or 'shift'")


def misreporting_measures(
    n_bidders: int,
    kappa: float,
    *,
    mode: MisreportingMode = "scale",
) -> Tuple[float, float, float, float]:
    lam_f = lambda_f_first_price(n_bidders)
    lam_i = informal_bid_multiplier(n_bidders, kappa, mode=mode)
    tilde_alpha = float(lam_i / lam_f - 1.0)
    alpha_additive = float(lam_i - lam_f)
    return float(lam_f), float(lam_i), tilde_alpha, alpha_additive

