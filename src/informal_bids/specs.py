from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Set, Tuple

import numpy as np

from .bounds import bounds_baseline
from .config import EPS_DIV
from .kernels import task_b_bounds_type_shift
from .misreporting import MisreportingMode
from .selection import p_select_baseline, p_select_type_shift
from .types import TaskBDataset, TaskBParams


def feature_gap23_ratio(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        raise ValueError("gap23_ratio requires at least 3 bids")
    s = np.sort(values)[::-1]
    b2, b3 = float(s[1]), float(s[2])
    denom = 0.5 * (abs(b2) + abs(b3)) + EPS_DIV
    return float((b2 - b3) / denom)


def feature_cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mu = float(np.mean(values))
    sd = float(np.std(values))
    return float(sd / (abs(mu) + EPS_DIV))


def feature_depth_exmax(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    s = np.sort(values)[::-1]
    if s.size < 2:
        raise ValueError("depth_exmax requires at least 2 bids")
    depth = float(np.mean(s[1:]))
    denom = abs(float(np.median(values))) + EPS_DIV
    return float(depth / denom)


class TaskBScreeningSpec(Protocol):
    name: str
    k_beta: int
    beta_names: List[str]
    depends_on_kappa_in_X: bool
    depends_on_params_in_bounds: Set[str]
    depends_on_params_in_select: Set[str]

    def compute_X(self, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode) -> np.ndarray:
        ...

    def compute_bounds(self, dataset: TaskBDataset, params: TaskBParams) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def selection_prob(
        self,
        b_star: np.ndarray,
        dataset: TaskBDataset,
        params: TaskBParams,
        *,
        misreporting_mode: MisreportingMode,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class _BaseSpec:
    name: str
    beta_names: List[str]
    depends_on_kappa_in_X: bool = False
    depends_on_params_in_bounds: Set[str] = frozenset()
    depends_on_params_in_select: Set[str] = frozenset()

    @property
    def k_beta(self) -> int:
        return int(len(self.beta_names))


@dataclass(frozen=True)
class LegacyInterceptSpec(_BaseSpec):
    name: str = "legacy_intercept"
    beta_names: List[str] = ("c",)

    def compute_X(self, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode) -> np.ndarray:
        return np.ones((dataset.N, 1), dtype=float)

    def compute_bounds(self, dataset: TaskBDataset, params: TaskBParams) -> Tuple[np.ndarray, np.ndarray]:
        L = np.empty(dataset.N, dtype=float)
        U = np.empty(dataset.N, dtype=float)
        for i in range(dataset.N):
            start = int(dataset.offsets[i])
            end = start + int(dataset.n_bidders[i])
            L[i], U[i] = bounds_baseline(dataset.bI[start:end], dataset.admitted[start:end])
        return L, U

    def selection_prob(
        self, b_star: np.ndarray, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode
    ) -> np.ndarray:
        return p_select_baseline(
            b_star,
            gamma=params.gamma,
            sigma_nu=params.sigma_nu,
            n_bidders=dataset.n_bidders,
            kappa=params.kappa,
            misreporting_mode=misreporting_mode,
        )


@dataclass(frozen=True)
class LegacyMomentsK4Spec(_BaseSpec):
    name: str = "legacy_moments_k4"
    beta_names: List[str] = ("c", "m1_top1", "m2_top2_avg", "m3_top3_avg")

    def compute_X(self, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode) -> np.ndarray:
        X = np.ones((dataset.N, 4), dtype=float)
        for i in range(dataset.N):
            start = int(dataset.offsets[i])
            end = start + int(dataset.n_bidders[i])
            b = np.sort(dataset.bI[start:end])[::-1]
            if b.size < 3:
                raise ValueError("legacy_moments_k4 requires J>=3")
            X[i, 1] = float(b[0])
            X[i, 2] = float(np.mean(b[:2]))
            X[i, 3] = float(np.mean(b[:3]))
        return X

    def compute_bounds(self, dataset: TaskBDataset, params: TaskBParams) -> Tuple[np.ndarray, np.ndarray]:
        return LegacyInterceptSpec().compute_bounds(dataset, params)

    def selection_prob(
        self, b_star: np.ndarray, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode
    ) -> np.ndarray:
        return LegacyInterceptSpec().selection_prob(b_star, dataset, params, misreporting_mode=misreporting_mode)


@dataclass(frozen=True)
class LegacyDepthK2Spec(_BaseSpec):
    name: str = "legacy_depth_k2"
    beta_names: List[str] = ("c", "depth_mean_23", "depth_gap_23")

    def compute_X(self, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode) -> np.ndarray:
        X = np.ones((dataset.N, 3), dtype=float)
        for i in range(dataset.N):
            start = int(dataset.offsets[i])
            end = start + int(dataset.n_bidders[i])
            b = np.sort(dataset.bI[start:end])[::-1]
            if b.size < 3:
                raise ValueError("legacy_depth_k2 requires J>=3")
            b2, b3 = float(b[1]), float(b[2])
            X[i, 1] = 0.5 * (b2 + b3)
            X[i, 2] = b2 - b3
        return X

    def compute_bounds(self, dataset: TaskBDataset, params: TaskBParams) -> Tuple[np.ndarray, np.ndarray]:
        return LegacyInterceptSpec().compute_bounds(dataset, params)

    def selection_prob(
        self, b_star: np.ndarray, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode
    ) -> np.ndarray:
        return LegacyInterceptSpec().selection_prob(b_star, dataset, params, misreporting_mode=misreporting_mode)


@dataclass(frozen=True)
class LegacyDepthK2RatioSpec(_BaseSpec):
    name: str = "legacy_depth_k2_ratio"
    beta_names: List[str] = ("c", "depth_mean_23", "depth_gap_23_ratio")

    def compute_X(self, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode) -> np.ndarray:
        X = np.ones((dataset.N, 3), dtype=float)
        for i in range(dataset.N):
            start = int(dataset.offsets[i])
            end = start + int(dataset.n_bidders[i])
            b = np.sort(dataset.bI[start:end])[::-1]
            if b.size < 3:
                raise ValueError("legacy_depth_k2_ratio requires J>=3")
            b2, b3 = float(b[1]), float(b[2])
            m1 = 0.5 * (b2 + b3)
            gap = b2 - b3
            X[i, 1] = m1
            X[i, 2] = gap / (abs(m1) + EPS_DIV)
        return X

    def compute_bounds(self, dataset: TaskBDataset, params: TaskBParams) -> Tuple[np.ndarray, np.ndarray]:
        return LegacyInterceptSpec().compute_bounds(dataset, params)

    def selection_prob(
        self, b_star: np.ndarray, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode
    ) -> np.ndarray:
        return LegacyInterceptSpec().selection_prob(b_star, dataset, params, misreporting_mode=misreporting_mode)


@dataclass(frozen=True)
class Cand1TypeSprSpec(_BaseSpec):
    name: str = "cand1_type_spr"
    beta_names: List[str] = ("c", "theta_type_shareS", "theta_spr_gap23_ratio")

    def compute_X(self, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode) -> np.ndarray:
        X = np.ones((dataset.N, 3), dtype=float)
        X[:, 1] = dataset.g_shareS
        for i in range(dataset.N):
            start = int(dataset.offsets[i])
            end = start + int(dataset.n_bidders[i])
            X[i, 2] = feature_gap23_ratio(dataset.bI[start:end])
        return X

    def compute_bounds(self, dataset: TaskBDataset, params: TaskBParams) -> Tuple[np.ndarray, np.ndarray]:
        return LegacyInterceptSpec().compute_bounds(dataset, params)

    def selection_prob(
        self, b_star: np.ndarray, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode
    ) -> np.ndarray:
        return LegacyInterceptSpec().selection_prob(b_star, dataset, params, misreporting_mode=misreporting_mode)


@dataclass(frozen=True)
class Cand2TypeSprDepthZSpec(_BaseSpec):
    name: str = "cand2_type_spr_depth_z"
    beta_names: List[str] = ("c", "theta_type_shareS", "theta_spr_cv", "theta_depth_exmax", "pi_Z")

    def compute_X(self, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode) -> np.ndarray:
        X = np.ones((dataset.N, 5), dtype=float)
        X[:, 1] = dataset.g_shareS
        X[:, 4] = dataset.Z
        for i in range(dataset.N):
            start = int(dataset.offsets[i])
            end = start + int(dataset.n_bidders[i])
            b = dataset.bI[start:end]
            X[i, 2] = feature_cv(b)
            X[i, 3] = feature_depth_exmax(b)
        return X

    def compute_bounds(self, dataset: TaskBDataset, params: TaskBParams) -> Tuple[np.ndarray, np.ndarray]:
        return LegacyInterceptSpec().compute_bounds(dataset, params)

    def selection_prob(
        self, b_star: np.ndarray, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode
    ) -> np.ndarray:
        return LegacyInterceptSpec().selection_prob(b_star, dataset, params, misreporting_mode=misreporting_mode)


@dataclass(frozen=True)
class Cand3TypeShiftAdmissionSpec(_BaseSpec):
    name: str = "cand3_type_shift_admission"
    beta_names: List[str] = ("c", "theta_type_shareS", "theta_spr_gap23_ratio")
    depends_on_params_in_bounds: Set[str] = frozenset({"delta"})
    depends_on_params_in_select: Set[str] = frozenset({"delta"})

    def compute_X(self, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode) -> np.ndarray:
        return Cand1TypeSprSpec().compute_X(dataset, params, misreporting_mode=misreporting_mode)

    def compute_bounds(self, dataset: TaskBDataset, params: TaskBParams) -> Tuple[np.ndarray, np.ndarray]:
        return task_b_bounds_type_shift(
            dataset.bI,
            dataset.admitted,
            dataset.T,
            dataset.offsets,
            dataset.n_bidders,
            float(params.delta),
        )

    def selection_prob(
        self, b_star: np.ndarray, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode
    ) -> np.ndarray:
        return p_select_type_shift(
            b_star,
            gamma=params.gamma,
            sigma_nu=params.sigma_nu,
            nS=dataset.nS,
            nF=dataset.nF,
            kappa=params.kappa,
            delta=float(params.delta),
            misreporting_mode=misreporting_mode,
        )


@dataclass(frozen=True)
class Cand4TypeSprPrecZSpec(_BaseSpec):
    name: str = "cand4_type_spr_prec_z"
    beta_names: List[str] = ("c", "theta_type_shareS", "theta_spr_gap23_ratio", "theta_prec_tick", "pi_Z")

    def compute_X(self, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode) -> np.ndarray:
        X = np.ones((dataset.N, 5), dtype=float)
        X[:, 1] = dataset.g_shareS
        X[:, 3] = dataset.p_prec
        X[:, 4] = dataset.Z
        for i in range(dataset.N):
            start = int(dataset.offsets[i])
            end = start + int(dataset.n_bidders[i])
            X[i, 2] = feature_gap23_ratio(dataset.bI[start:end])
        return X

    def compute_bounds(self, dataset: TaskBDataset, params: TaskBParams) -> Tuple[np.ndarray, np.ndarray]:
        return LegacyInterceptSpec().compute_bounds(dataset, params)

    def selection_prob(
        self, b_star: np.ndarray, dataset: TaskBDataset, params: TaskBParams, *, misreporting_mode: MisreportingMode
    ) -> np.ndarray:
        return LegacyInterceptSpec().selection_prob(b_star, dataset, params, misreporting_mode=misreporting_mode)


TASKB_SPECS: Dict[str, TaskBScreeningSpec] = {
    "legacy_intercept": LegacyInterceptSpec(),
    "legacy_moments_k4": LegacyMomentsK4Spec(),
    "legacy_depth_k2": LegacyDepthK2Spec(),
    "legacy_depth_k2_ratio": LegacyDepthK2RatioSpec(),
    "cand1_type_spr": Cand1TypeSprSpec(),
    "cand2_type_spr_depth_z": Cand2TypeSprDepthZSpec(),
    "cand3_type_shift_admission": Cand3TypeShiftAdmissionSpec(),
    "cand4_type_spr_prec_z": Cand4TypeSprPrecZSpec(),
}
