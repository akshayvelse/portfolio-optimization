"""Portfolio optimization algorithms and covariance shrinkage estimators.

Implements four systematic asset allocation strategies:
1. Equal-Weighted (1/N benchmark)
2. Market-Capitalization Weighted
3. Unconstrained Mean-Variance Optimization (Sample covariance, prone to corner solutions)
4. Constrained Institutional Allocation (Ledoit-Wolf shrinkage, 15% position cap, turnover regularizer)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def compute_ledoit_wolf_covariance(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Calculate Ledoit-Wolf shrinkage covariance matrix with constant correlation target.

    Formula:
        Sigma_LW = delta * F + (1 - delta) * S
    where:
        S = sample covariance matrix
        F = constant correlation shrinkage target matrix
        delta = optimal shrinkage intensity parameter in [0, 1]
    Shrinks sample estimation error and eliminates negative eigenvalue instability.
    """
    T, N = returns.shape
    X = returns - np.mean(returns, axis=0)
    sample_cov = (X.T @ X) / T

    # Variance vector and standard deviations
    variances = np.diag(sample_cov)
    std_devs = np.sqrt(variances)

    # Sample correlation matrix
    outer_std = np.outer(std_devs, std_devs)
    sample_corr = sample_cov / outer_std

    # Average pairwise correlation across non-diagonal elements
    r_bar = float((np.sum(sample_corr) - N) / (N * (N - 1)))

    # Shrinkage target matrix F: individual sample variances with constant correlation off-diagonals
    target_F = r_bar * outer_std
    np.fill_diagonal(target_F, variances)

    # Estimation of optimal shrinkage intensity delta
    # Sum of asymptotic variances of sample covariance entries (pi_mat)
    X2 = X**2
    sample_cov_sq = sample_cov**2
    pi_mat = ((X2.T @ X2) / T) - sample_cov_sq
    pi_hat = float(np.sum(pi_mat))

    # Frobenius norm distance squared between sample covariance and target
    gamma_hat = float(np.linalg.norm(sample_cov - target_F, "fro") ** 2)

    # Compute optimal shrinkage parameter
    if gamma_hat == 0.0:
        delta = 0.0
    else:
        kappa = pi_hat / gamma_hat
        delta = max(0.0, min(1.0, kappa / float(T)))

    shrunk_cov = delta * target_F + (1.0 - delta) * sample_cov

    # Symmetrize and ensure positive semi-definiteness
    shrunk_cov = 0.5 * (shrunk_cov + shrunk_cov.T)
    min_eig = np.min(np.linalg.eigvalsh(shrunk_cov))
    if min_eig < 1e-8:
        shrunk_cov += (abs(min_eig) + 1e-6) * np.eye(N)

    return shrunk_cov, delta


def optimize_equal_weight(n_assets: int) -> np.ndarray:
    """Strategy 1: 1/N equal weighting allocation."""
    return np.ones(n_assets) / float(n_assets)


def optimize_market_cap_weight(market_caps: pd.Series) -> np.ndarray:
    """Strategy 2: Value-weighted allocation proportional to market capitalization."""
    mc_values = market_caps.values.astype(float)
    total_mc = np.sum(mc_values)
    if total_mc <= 0:
        return np.ones(len(mc_values)) / float(len(mc_values))
    return mc_values / total_mc


def optimize_unconstrained_mvo(
    expected_returns: np.ndarray,
    sample_covariance: np.ndarray,
    risk_free_rate: float = 0.030,
) -> np.ndarray:
    """Strategy 3: Unconstrained long-only Mean-Variance Optimization (Max Sharpe).

    Prone to Markowitz error-maximization: concentrates into corner solutions
    with 0% weights across most assets and extreme allocations in noisy outliers.
    """
    n_assets = len(expected_returns)
    w0 = np.ones(n_assets) / float(n_assets)

    def negative_sharpe(w: np.ndarray) -> float:
        port_ret = float(np.dot(w, expected_returns))
        port_var = float(np.dot(w, np.dot(sample_covariance, w)))
        port_vol = np.sqrt(max(port_var, 1e-8))
        return -(port_ret - risk_free_rate) / port_vol

    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        negative_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if res.success:
        weights = res.x
    else:
        weights = w0

    # Clean small precision float dust and re-normalize
    weights = np.clip(weights, 0.0, 1.0)
    return weights / np.sum(weights)


def optimize_constrained_institutional(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    previous_weights: np.ndarray | None = None,
    risk_free_rate: float = 0.030,
    max_position_cap: float = 0.150,
    turnover_penalty_lambda: float = 0.080,
) -> np.ndarray:
    """Strategy 4: Institutional regularized portfolio allocation.

    Features:
    - Ledoit-Wolf shrinkage covariance inputs
    - Strict 15% maximum single-position weight ceiling
    - L1 turnover penalty in objective to penalize excessive rebalancing drag
    """
    n_assets = len(expected_returns)
    w0 = (
        previous_weights.copy()
        if previous_weights is not None
        else np.ones(n_assets) / float(n_assets)
    )

    def penalized_objective(w: np.ndarray) -> float:
        port_ret = float(np.dot(w, expected_returns))
        port_var = float(np.dot(w, np.dot(covariance_matrix, w)))
        port_vol = np.sqrt(max(port_var, 1e-8))
        base_sharpe = (port_ret - risk_free_rate) / port_vol

        # L1 turnover penalty
        if previous_weights is not None:
            turnover = float(np.sum(np.abs(w - previous_weights)))
        else:
            turnover = 0.0

        return -(base_sharpe - turnover_penalty_lambda * turnover)

    bounds = [(0.0, max_position_cap) for _ in range(n_assets)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        penalized_objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if res.success:
        weights = res.x
    else:
        weights = np.clip(w0, 0.0, max_position_cap)
        weights /= np.sum(weights)

    weights = np.clip(weights, 0.0, max_position_cap)
    return weights / np.sum(weights)


if __name__ == "__main__":
    n = 12
    fake_ret = np.random.randn(252, n) * 0.015
    cov_lw, delta = compute_ledoit_wolf_covariance(fake_ret)
    mu = np.mean(fake_ret, axis=0) * 252

    w_eq = optimize_equal_weight(n)
    w_mvo = optimize_unconstrained_mvo(mu, cov_lw)
    w_inst = optimize_constrained_institutional(mu, cov_lw)

    print("Ledoit-Wolf shrinkage intensity delta:", round(delta, 4))
    print("MVO max weight:", round(np.max(w_mvo), 4))
    print("Inst max weight (cap 0.15):", round(np.max(w_inst), 4))
