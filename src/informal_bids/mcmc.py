from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import ndtr, ndtri

from .kernels import (
    HAS_NUMBA,
    NUMBA_JIT_ENABLED,
    task_b_log_selection_sum,
    task_b_logpost_gamma,
    task_b_logpost_kappa,
    task_b_sum_sq_eta,
    task_b_sum_sq_v,
)
from .misreporting import MisreportingMode, informal_bid_multiplier, misreporting_measures
from .selection import p_select_baseline_scalar, p_select_type_shift_scalar
from .specs import TASKB_SPECS, TaskBScreeningSpec
from .types import TaskBDataset, TaskBParams, pack_task_b_dataset


@dataclass
class MCMCConfig:
    n_iterations: int = 15000
    burn_in: int = 7500
    thinning: int = 10
    n_chains: int = 2

    # Stage control: 1=fixed (sigma_nu,sigma_eta), 2=estimate sigma_eta, 3=estimate sigma_nu and sigma_eta
    stage: int = 1
    sigma_nu_fixed: float = 0.2
    sigma_eta_fixed: float = 0.1

    # Priors / proposals
    beta_prior_mean: float = 1.4
    beta_prior_std: float = 0.5

    sigma_omega_init: float = 0.1
    sigma_omega_prior_a: float = 2.0
    sigma_omega_prior_b: float = 0.1

    gamma_prior_mean: float = 1.3
    gamma_prior_std: float = 0.5
    gamma_prop_sd: float = 0.02

    kappa_prior_mean: float = 0.0
    kappa_prior_std: float = 0.5
    kappa_init: float = 0.0
    kappa_prop_sd: float = 0.1

    sigma_nu_prior_a: float = 2.0
    sigma_nu_prior_b: float = 0.1
    sigma_eta_prior_a: float = 2.0
    sigma_eta_prior_b: float = 0.1

    delta_prior_mean: float = 0.0
    delta_prior_std: float = 0.2
    delta_init: float = 0.0
    delta_prop_sd: float = 0.05

    # Performance
    use_numba: bool = True
    log_every: int = 10000


def sample_truncnorm_vec(xb: np.ndarray, sigma: float, L: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Vectorized truncated-normal sampler via inverse CDF (fast, no Python loops)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    xb = np.asarray(xb, dtype=float)
    L = np.asarray(L, dtype=float)
    U = np.asarray(U, dtype=float)
    if xb.shape != L.shape or xb.shape != U.shape:
        raise ValueError("xb, L, U must have the same shape")

    a = (L - xb) / float(sigma)
    b = (U - xb) / float(sigma)
    Fa = ndtr(a)
    Fb = ndtr(b)
    u = Fa + np.random.rand(xb.size) * (Fb - Fa)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    z = ndtri(u)
    return xb + float(sigma) * z


def invgamma_rvs_scalar(shape: float, *, scale: float) -> float:
    """Sample from InvGamma(shape, scale) using NumPy (faster than scipy.stats in a tight loop)."""
    if shape <= 0.0 or scale <= 0.0:
        raise ValueError("shape and scale must be positive")
    return float(scale / np.random.gamma(shape=shape, scale=1.0))


def gelman_rubin(chains: List[np.ndarray]) -> np.ndarray:
    """Gelman-Rubin R-hat for chains shaped (n_iter, d)."""
    if len(chains) < 2:
        return np.full((chains[0].shape[1],), np.nan, dtype=float)
    m = len(chains)
    n = min(c.shape[0] for c in chains)
    if n < 2:
        return np.full((chains[0].shape[1],), np.nan, dtype=float)
    xs = np.stack([c[:n] for c in chains], axis=0)  # (m,n,d)
    chain_means = xs.mean(axis=1)  # (m,d)
    grand_mean = chain_means.mean(axis=0)  # (d,)
    B = n * ((chain_means - grand_mean) ** 2).sum(axis=0) / (m - 1)
    W = xs.var(axis=1, ddof=1).mean(axis=0)
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / W)
    rhat[~np.isfinite(rhat)] = np.nan
    return rhat


def collinearity_diagnostics(X: np.ndarray) -> Dict:
    if X.size == 0:
        return {}
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if k <= 1 or n <= 2:
        return {"n_obs": n, "n_features": k, "n_features_ex_intercept": max(0, k - 1)}

    X0 = X[:, 1:]
    corr = np.corrcoef(X0, rowvar=False)
    max_abs_corr = float(np.max(np.abs(corr[np.triu_indices_from(corr, k=1)]))) if X0.shape[1] > 1 else 0.0
    cond = float(np.linalg.cond(X0))

    vifs = []
    for j in range(X0.shape[1]):
        y = X0[:, j]
        Xj = np.delete(X0, j, axis=1)
        if Xj.shape[1] == 0:
            vifs.append(np.nan)
            continue
        coef, *_ = np.linalg.lstsq(Xj, y, rcond=None)
        resid = y - Xj @ coef
        ssr = float(np.sum(resid * resid))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ssr / sst if sst > 0 else 0.0
        vif = float(1.0 / max(1e-12, (1.0 - r2)))
        vifs.append(vif)

    max_vif = float(np.nanmax(vifs)) if vifs else np.nan
    return {
        "n_obs": n,
        "n_features": k,
        "n_features_ex_intercept": k - 1,
        "max_abs_corr": max_abs_corr,
        "condition_number": cond,
        "vifs": vifs,
        "max_vif": max_vif,
    }


class TaskBMHSampler:
    def __init__(
        self,
        auctions,
        *,
        spec_name: str,
        config: MCMCConfig,
        misreporting_mode: MisreportingMode = "scale",
    ):
        if not HAS_NUMBA:
            raise RuntimeError("Numba is required for this branch (performance-critical).")
        if not NUMBA_JIT_ENABLED:
            raise RuntimeError(
                "Numba is installed but JIT is disabled (NUMBA_DISABLE_JIT=1). "
                "This will be extremely slow; unset NUMBA_DISABLE_JIT to run this branch."
            )
        if not getattr(config, "use_numba", True):
            raise ValueError("use_numba must be True on this branch.")

        self.dataset: TaskBDataset = pack_task_b_dataset(auctions)
        if spec_name not in TASKB_SPECS:
            raise ValueError(f"Unknown spec_name '{spec_name}'")
        self.spec: TaskBScreeningSpec = TASKB_SPECS[spec_name]
        self.config = config
        self.misreporting_mode = misreporting_mode

        self._mode_flag = 0 if self.misreporting_mode == "scale" else 1
        self._use_type_shift = 1 if self.spec.name == "cand3_type_shift_admission" else 0

    def _get_beta_prior(self) -> Tuple[np.ndarray, np.ndarray]:
        k = self.spec.k_beta
        mean = np.atleast_1d(self.config.beta_prior_mean).astype(float)
        std = np.atleast_1d(self.config.beta_prior_std).astype(float)
        if mean.size == 1:
            if k == 1:
                mean = np.array([mean.item()], dtype=float)
            else:
                mean_full = np.zeros(k, dtype=float)
                mean_full[0] = mean.item()
                mean = mean_full
        if std.size == 1:
            std = np.full(k, std.item())
        if mean.size != k or std.size != k:
            raise ValueError("beta prior has wrong length")
        return mean, std

    def _compute_v_slices(self, *, kappa: float) -> List[np.ndarray]:
        v_slices = []
        for i in range(self.dataset.N):
            n = int(self.dataset.n_bidders[i])
            lam_i = informal_bid_multiplier(n, kappa, mode=self.misreporting_mode)
            start = int(self.dataset.offsets[i])
            end = start + n
            v_slices.append(self.dataset.bI[start:end] / lam_i)
        return v_slices

    def _selection_prob_scalar(self, *, b_star: float, params: TaskBParams, i: int) -> float:
        if self.spec.name == "cand3_type_shift_admission":
            return p_select_type_shift_scalar(
                b_star,
                gamma=params.gamma,
                sigma_nu=params.sigma_nu,
                nS=int(self.dataset.nS[i]),
                nF=int(self.dataset.nF[i]),
                kappa=params.kappa,
                delta=params.delta,
                misreporting_mode=self.misreporting_mode,
            )
        return p_select_baseline_scalar(
            b_star,
            gamma=params.gamma,
            sigma_nu=params.sigma_nu,
            n_bidders=int(self.dataset.n_bidders[i]),
            kappa=params.kappa,
            misreporting_mode=self.misreporting_mode,
        )

    def _logpost_gamma(self, gamma: float, *, params: TaskBParams, b_star: np.ndarray) -> float:
        return float(
            task_b_logpost_gamma(
                float(gamma),
                float(self.config.gamma_prior_mean),
                float(self.config.gamma_prior_std),
                b_star.astype(np.float64, copy=False),
                self.dataset.bI.astype(np.float64, copy=False),
                self.dataset.offsets.astype(np.int64, copy=False),
                self.dataset.n_bidders.astype(np.int64, copy=False),
                self.dataset.nS.astype(np.int64, copy=False),
                self.dataset.nF.astype(np.int64, copy=False),
                float(params.sigma_nu),
                float(params.kappa),
                float(params.delta),
                int(self._mode_flag),
                int(self._use_type_shift),
            )
        )

    def _logpost_kappa(self, kappa: float, *, params: TaskBParams, b_star: np.ndarray, X: np.ndarray) -> float:
        lp = float(
            task_b_logpost_kappa(
                float(kappa),
                float(self.config.kappa_prior_mean),
                float(self.config.kappa_prior_std),
                float(params.gamma),
                float(params.sigma_nu),
                float(params.sigma_eta),
                b_star.astype(np.float64, copy=False),
                self.dataset.bI.astype(np.float64, copy=False),
                self.dataset.bF.astype(np.float64, copy=False),
                self.dataset.admitted.astype(np.bool_, copy=False),
                self.dataset.offsets.astype(np.int64, copy=False),
                self.dataset.n_bidders.astype(np.int64, copy=False),
                self.dataset.lambda_f.astype(np.float64, copy=False),
                self.dataset.nS.astype(np.int64, copy=False),
                self.dataset.nF.astype(np.int64, copy=False),
                float(params.delta),
                int(self._mode_flag),
                int(self._use_type_shift),
            )
        )

        # If X depends on kappa, include cutoff regression term in the kappa MH ratio.
        if getattr(self.spec, "depends_on_kappa_in_X", False):
            resid = b_star - X @ params.beta
            lp += -0.5 * float(np.sum((resid / params.sigma_omega) ** 2))
            lp += -float(self.dataset.N) * float(np.log(params.sigma_omega))

        return float(lp)

    def run_chain(self, chain_id: int) -> Dict:
        stage = int(self.config.stage)
        if stage not in (1, 2, 3):
            raise ValueError("stage must be 1, 2, or 3")

        beta_mean, beta_std = self._get_beta_prior()
        V0_inv = np.diag(1.0 / (beta_std ** 2))
        V0_inv_beta_mean = V0_inv @ beta_mean

        beta = beta_mean.copy()
        sigma_omega = float(self.config.sigma_omega_init)
        gamma = float(self.config.gamma_prior_mean)
        kappa = float(self.config.kappa_init)
        delta = float(self.config.delta_init)

        sigma_nu = float(self.config.sigma_nu_fixed)
        sigma_eta = float(self.config.sigma_eta_fixed)

        n_iter = int(self.config.n_iterations)
        k = int(self.spec.k_beta)

        beta_chain = np.zeros((n_iter, k), dtype=float)
        gamma_chain = np.zeros(n_iter, dtype=float)
        kappa_chain = np.zeros(n_iter, dtype=float)
        delta_chain = np.zeros(n_iter, dtype=float)
        sigma_omega_chain = np.zeros(n_iter, dtype=float)
        sigma_nu_chain = np.zeros(n_iter, dtype=float)
        sigma_eta_chain = np.zeros(n_iter, dtype=float)

        params = TaskBParams(beta=beta, sigma_omega=sigma_omega, gamma=gamma, kappa=kappa, sigma_nu=sigma_nu, sigma_eta=sigma_eta, delta=delta)
        X = self.spec.compute_X(self.dataset, params, misreporting_mode=self.misreporting_mode)

        bounds_depend_on_params = bool(getattr(self.spec, "depends_on_params_in_bounds", set()))
        if bounds_depend_on_params:
            L, U = self.spec.compute_bounds(self.dataset, params)
        else:
            L, U = self.spec.compute_bounds(self.dataset, params)

        xb0 = X @ beta
        b_star = sample_truncnorm_vec(xb0, float(sigma_omega), L, U)

        XtX = X.T @ X if self.dataset.N > 0 else np.empty((k, k), dtype=float)

        # Cache selection probabilities for the b_star MH step to avoid recomputing p_old each iteration.
        # This cache is only valid for the current (b_star, gamma, sigma_nu, kappa, delta).
        p_sel = self.spec.selection_prob(b_star, self.dataset, params, misreporting_mode=self.misreporting_mode)
        p_sel_dirty = False

        a_omega = float(self.config.sigma_omega_prior_a)
        b_omega = float(self.config.sigma_omega_prior_b)
        a_nu = float(self.config.sigma_nu_prior_a)
        b_nu = float(self.config.sigma_nu_prior_b)
        a_eta = float(self.config.sigma_eta_prior_a)
        b_eta = float(self.config.sigma_eta_prior_b)

        gamma_prop_sd = float(self.config.gamma_prop_sd)
        kappa_prop_sd = float(self.config.kappa_prop_sd)
        sd_delta = float(self.config.delta_prior_std)
        delta_prop_sd = float(self.config.delta_prop_sd)

        n_bstar_accept = 0
        n_bstar_total = 0
        n_gamma_accept = 0
        n_kappa_accept = 0
        n_delta_accept = 0

        for t in range(n_iter):
            params = TaskBParams(
                beta=beta,
                sigma_omega=sigma_omega,
                gamma=gamma,
                kappa=kappa,
                sigma_nu=sigma_nu,
                sigma_eta=sigma_eta,
                delta=delta,
            )

            if bounds_depend_on_params:
                L, U = self.spec.compute_bounds(self.dataset, params)

            # Step 1: update b_star via independence MH
            if p_sel_dirty:
                p_sel = self.spec.selection_prob(b_star, self.dataset, params, misreporting_mode=self.misreporting_mode)
                p_sel_dirty = False
            xb = X @ beta
            b_prop = sample_truncnorm_vec(xb, float(sigma_omega), L, U)

            p_prop = self.spec.selection_prob(b_prop, self.dataset, params, misreporting_mode=self.misreporting_mode)
            alpha = p_sel / p_prop
            u = np.random.rand(self.dataset.N)
            accept = (alpha >= 1.0) | (u < alpha)

            n_bstar_total += int(self.dataset.N)
            n_bstar_accept += int(np.sum(accept))
            b_star[accept] = b_prop[accept]
            p_sel[accept] = p_prop[accept]

            # Step 2: update beta
            if self.dataset.N > 0:
                V_post = np.linalg.inv(V0_inv + XtX / (sigma_omega ** 2))
                beta_post = V_post @ (V0_inv_beta_mean + (X.T @ b_star) / (sigma_omega ** 2))
                beta = np.random.multivariate_normal(beta_post, V_post)

            # Step 3: update sigma_omega
            resid = b_star - X @ beta
            a_post = a_omega + 0.5 * self.dataset.N
            b_post = b_omega + 0.5 * float(np.sum(resid * resid))
            sigma_omega = float(np.sqrt(invgamma_rvs_scalar(a_post, scale=b_post)))

            # Step 4: update gamma
            gamma_prop = float(gamma + np.random.normal(0.0, gamma_prop_sd))
            lp_old = self._logpost_gamma(gamma, params=params, b_star=b_star)
            params_gamma_prop = TaskBParams(beta=beta, sigma_omega=sigma_omega, gamma=gamma_prop, kappa=kappa, sigma_nu=sigma_nu, sigma_eta=sigma_eta, delta=delta)
            lp_prop = self._logpost_gamma(gamma_prop, params=params_gamma_prop, b_star=b_star)
            if lp_prop >= lp_old or np.log(np.random.rand()) < (lp_prop - lp_old):
                gamma = gamma_prop
                p_sel_dirty = True
                n_gamma_accept += 1

            # Step 5: update sigma_nu (stage 3)
            if stage == 3:
                ss, n_v = task_b_sum_sq_v(
                    self.dataset.bI.astype(np.float64, copy=False),
                    self.dataset.offsets.astype(np.int64, copy=False),
                    self.dataset.n_bidders.astype(np.int64, copy=False),
                    float(gamma),
                    float(kappa),
                    int(self._mode_flag),
                )
                if n_v > 0:
                    a_post = a_nu + 0.5 * n_v
                    b_post = b_nu + 0.5 * ss
                    sigma_nu_prop = float(np.sqrt(invgamma_rvs_scalar(a_post, scale=b_post)))

                    lp_sel_old = task_b_log_selection_sum(
                        b_star.astype(np.float64, copy=False),
                        self.dataset.n_bidders.astype(np.int64, copy=False),
                        self.dataset.nS.astype(np.int64, copy=False),
                        self.dataset.nF.astype(np.int64, copy=False),
                        float(gamma),
                        float(sigma_nu),
                        float(kappa),
                        float(delta),
                        int(self._mode_flag),
                        int(self._use_type_shift),
                    )
                    lp_sel_prop = task_b_log_selection_sum(
                        b_star.astype(np.float64, copy=False),
                        self.dataset.n_bidders.astype(np.int64, copy=False),
                        self.dataset.nS.astype(np.int64, copy=False),
                        self.dataset.nF.astype(np.int64, copy=False),
                        float(gamma),
                        float(sigma_nu_prop),
                        float(kappa),
                        float(delta),
                        int(self._mode_flag),
                        int(self._use_type_shift),
                    )
                    log_alpha = float(lp_sel_old - lp_sel_prop)
                    if log_alpha >= 0.0 or np.log(np.random.rand()) < log_alpha:
                        sigma_nu = sigma_nu_prop
                        p_sel_dirty = True

            # Step 6: update sigma_eta (stage 2/3)
            if stage in (2, 3):
                ss_eta, n_eta = task_b_sum_sq_eta(
                    self.dataset.bI.astype(np.float64, copy=False),
                    self.dataset.bF.astype(np.float64, copy=False),
                    self.dataset.admitted.astype(np.bool_, copy=False),
                    self.dataset.offsets.astype(np.int64, copy=False),
                    self.dataset.n_bidders.astype(np.int64, copy=False),
                    self.dataset.lambda_f.astype(np.float64, copy=False),
                    float(kappa),
                    int(self._mode_flag),
                )
                if n_eta > 0:
                    a_post = a_eta + 0.5 * float(n_eta)
                    b_post = b_eta + 0.5 * float(ss_eta)
                    sigma_eta = float(np.sqrt(invgamma_rvs_scalar(a_post, scale=b_post)))

            # Step 7: update kappa
            kappa_prop = float(kappa + np.random.normal(0.0, kappa_prop_sd))
            params_k = TaskBParams(beta=beta, sigma_omega=sigma_omega, gamma=gamma, kappa=kappa, sigma_nu=sigma_nu, sigma_eta=sigma_eta, delta=delta)
            params_kp = TaskBParams(beta=beta, sigma_omega=sigma_omega, gamma=gamma, kappa=kappa_prop, sigma_nu=sigma_nu, sigma_eta=sigma_eta, delta=delta)

            X_prop = X
            if getattr(self.spec, "depends_on_kappa_in_X", False):
                X_prop = self.spec.compute_X(self.dataset, params_kp, misreporting_mode=self.misreporting_mode)
            lp_old = self._logpost_kappa(kappa, params=params_k, b_star=b_star, X=X)
            lp_prop = self._logpost_kappa(kappa_prop, params=params_kp, b_star=b_star, X=X_prop)
            if lp_prop >= lp_old or np.log(np.random.rand()) < (lp_prop - lp_old):
                kappa = kappa_prop
                X = X_prop
                if self.dataset.N > 0 and getattr(self.spec, "depends_on_kappa_in_X", False):
                    XtX = X.T @ X
                p_sel_dirty = True
                n_kappa_accept += 1

            # Step 8: update delta (Candidate 3)
            if self.spec.name == "cand3_type_shift_admission":
                d0 = float(self.config.delta_prior_mean)
                delta_prop = float(delta + np.random.normal(0.0, delta_prop_sd))

                def logpost_delta(d: float) -> float:
                    lp = -0.5 * ((d - d0) ** 2) / (sd_delta * sd_delta) - np.log(sd_delta)
                    params_d = TaskBParams(beta=beta, sigma_omega=sigma_omega, gamma=gamma, kappa=kappa, sigma_nu=sigma_nu, sigma_eta=sigma_eta, delta=d)
                    Ld, Ud = self.spec.compute_bounds(self.dataset, params_d)
                    if np.any(b_star < Ld) or np.any(b_star > Ud) or np.any(Ld > Ud):
                        return -np.inf
                    lp_sel = task_b_log_selection_sum(
                        b_star.astype(np.float64, copy=False),
                        self.dataset.n_bidders.astype(np.int64, copy=False),
                        self.dataset.nS.astype(np.int64, copy=False),
                        self.dataset.nF.astype(np.int64, copy=False),
                        float(gamma),
                        float(sigma_nu),
                        float(kappa),
                        float(d),
                        int(self._mode_flag),
                        1,
                    )
                    lp += -float(lp_sel)
                    return float(lp)

                lp_old = logpost_delta(delta)
                lp_prop = logpost_delta(delta_prop)
                if lp_prop >= lp_old or np.log(np.random.rand()) < (lp_prop - lp_old):
                    delta = delta_prop
                    p_sel_dirty = True
                    n_delta_accept += 1

            beta_chain[t] = beta
            gamma_chain[t] = gamma
            kappa_chain[t] = kappa
            delta_chain[t] = delta
            sigma_omega_chain[t] = sigma_omega
            sigma_nu_chain[t] = sigma_nu
            sigma_eta_chain[t] = sigma_eta

            log_every = int(getattr(self.config, "log_every", 0) or 0)
            if log_every > 0 and (t + 1) % log_every == 0:
                acc_b = n_bstar_accept / float(max(1, n_bstar_total))
                acc_g = n_gamma_accept / float(t + 1)
                acc_k = n_kappa_accept / float(t + 1)
                acc_d = n_delta_accept / float(t + 1) if self.spec.name == "cand3_type_shift_admission" else 0.0
                tilde_alpha = misreporting_measures(int(self.dataset.n_bidders[0]), float(kappa), mode=self.misreporting_mode)[2]
                print(
                    f"  Iter {t+1}: gamma={gamma:.3f}, kappa={kappa:.3f}, tilde_alpha={tilde_alpha:.3f}, "
                    f"c0={beta[0]:.3f}, sigma_omega={sigma_omega:.3f}, delta={delta:.3f}, "
                    f"acc(b*)={acc_b:.2f}, acc(g)={acc_g:.2f}, acc(k)={acc_k:.2f}, acc(d)={acc_d:.2f}",
                    flush=True,
                )

        return {
            "beta": beta_chain,
            "gamma": gamma_chain,
            "kappa": kappa_chain,
            "delta": delta_chain,
            "sigma_omega": sigma_omega_chain,
            "sigma_nu": sigma_nu_chain,
            "sigma_eta": sigma_eta_chain,
            "acc_bstar": n_bstar_accept / float(max(1, n_bstar_total)),
            "acc_gamma": n_gamma_accept / float(n_iter),
            "acc_kappa": n_kappa_accept / float(n_iter),
            "acc_delta": n_delta_accept / float(n_iter) if self.spec.name == "cand3_type_shift_admission" else 0.0,
        }

    def run(self) -> Dict:
        chains = [self.run_chain(i) for i in range(int(self.config.n_chains))]

        burn = int(self.config.burn_in)
        thin = int(self.config.thinning)

        beta_slices = [c["beta"][burn::thin] for c in chains]
        gamma_slices = [c["gamma"][burn::thin] for c in chains]
        kappa_slices = [c["kappa"][burn::thin] for c in chains]
        delta_slices = [c["delta"][burn::thin] for c in chains]
        sigma_omega_slices = [c["sigma_omega"][burn::thin] for c in chains]

        beta_samples = np.concatenate(beta_slices, axis=0) if beta_slices else np.empty((0, self.spec.k_beta))
        gamma_samples = np.concatenate(gamma_slices, axis=0) if gamma_slices else np.empty((0,))
        kappa_samples = np.concatenate(kappa_slices, axis=0) if kappa_slices else np.empty((0,))
        delta_samples = np.concatenate(delta_slices, axis=0) if delta_slices else np.empty((0,))
        sigma_omega_samples = np.concatenate(sigma_omega_slices, axis=0) if sigma_omega_slices else np.empty((0,))

        rhat_beta = gelman_rubin([c["beta"][burn:] for c in chains])
        rhat_gamma = float(gelman_rubin([c["gamma"][burn:].reshape(-1, 1) for c in chains])[0])
        rhat_kappa = float(gelman_rubin([c["kappa"][burn:].reshape(-1, 1) for c in chains])[0])
        rhat_delta = float(gelman_rubin([c["delta"][burn:].reshape(-1, 1) for c in chains])[0])

        params_mean = TaskBParams(
            beta=np.mean(beta_samples, axis=0) if beta_samples.size else np.zeros(self.spec.k_beta),
            sigma_omega=float(np.mean(sigma_omega_samples)) if sigma_omega_samples.size else float(self.config.sigma_omega_init),
            gamma=float(np.mean(gamma_samples)) if gamma_samples.size else float(self.config.gamma_prior_mean),
            kappa=float(np.mean(kappa_samples)) if kappa_samples.size else float(self.config.kappa_init),
            sigma_nu=float(self.config.sigma_nu_fixed),
            sigma_eta=float(self.config.sigma_eta_fixed),
            delta=float(np.mean(delta_samples)) if delta_samples.size else float(self.config.delta_init),
        )
        X = self.spec.compute_X(self.dataset, params_mean, misreporting_mode=self.misreporting_mode)
        col_diag = collinearity_diagnostics(X)

        return {
            "spec_name": self.spec.name,
            "beta_names": list(self.spec.beta_names),
            "beta_samples": beta_samples,
            "gamma_samples": gamma_samples,
            "kappa_samples": kappa_samples,
            "delta_samples": delta_samples,
            "sigma_omega_samples": sigma_omega_samples,
            "all_chains": chains,
            "rhat_beta": rhat_beta,
            "rhat_gamma": rhat_gamma,
            "rhat_kappa": rhat_kappa,
            "rhat_delta": rhat_delta,
            "acc_bstar": float(np.mean([c["acc_bstar"] for c in chains])),
            "acc_gamma": float(np.mean([c["acc_gamma"] for c in chains])),
            "acc_kappa": float(np.mean([c["acc_kappa"] for c in chains])),
            "acc_delta": float(np.mean([c["acc_delta"] for c in chains])),
            "collinearity_diagnostics": col_diag,
        }
