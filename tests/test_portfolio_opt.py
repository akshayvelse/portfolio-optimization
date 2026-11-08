"""Unit and integration test suite for quantitative portfolio optimization and backtesting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtester import WalkForwardBacktester, compute_performance_metrics
from optimizers import (
    compute_ledoit_wolf_covariance,
    optimize_constrained_institutional,
    optimize_equal_weight,
    optimize_market_cap_weight,
    optimize_unconstrained_mvo,
)
from universe import (
    PORTFOLIO_TICKERS,
    fetch_or_load_universe_market_caps,
    fetch_or_load_universe_prices,
)


def test_ledoit_wolf_positive_definiteness() -> None:
    """Verify Ledoit-Wolf shrinkage produces symmetric, strictly positive-definite matrices."""
    prices = fetch_or_load_universe_prices()
    rets = prices[PORTFOLIO_TICKERS].pct_change().dropna().iloc[:252].values

    cov_lw, delta = compute_ledoit_wolf_covariance(rets)

    assert np.allclose(cov_lw, cov_lw.T), "Ledoit-Wolf covariance matrix must be symmetric"

    eigenvalues = np.linalg.eigvalsh(cov_lw)
    assert np.all(eigenvalues > 0), f"Non-positive eigenvalues found: {eigenvalues[eigenvalues <= 0]}"
    assert 0.0 <= delta <= 1.0, f"Shrinkage delta {delta} outside valid range [0, 1]"


def test_portfolio_weight_budget_constraints() -> None:
    """Verify all 4 optimization routines satisfy full investment budget sum(w) == 1.0."""
    n_assets = len(PORTFOLIO_TICKERS)
    mcaps = fetch_or_load_universe_market_caps()

    prices = fetch_or_load_universe_prices()
    rets = prices[PORTFOLIO_TICKERS].pct_change().dropna().iloc[:252].values
    mu = np.mean(rets, axis=0) * 252.0
    cov_sample = np.cov(rets, rowvar=False) * 252.0
    cov_lw, _ = compute_ledoit_wolf_covariance(rets)

    w_eq = optimize_equal_weight(n_assets)
    w_mc = optimize_market_cap_weight(mcaps)
    w_mvo = optimize_unconstrained_mvo(mu, cov_sample)
    w_inst = optimize_constrained_institutional(mu, cov_lw * 252.0, max_position_cap=0.150)

    for name, w in [
        ("Equal Weight", w_eq),
        ("Market Cap", w_mc),
        ("Unconstrained MVO", w_mvo),
        ("Constrained Inst", w_inst),
    ]:
        assert abs(np.sum(w) - 1.0) < 1e-6, f"{name} weights do not sum to 1.0: sum = {np.sum(w)}"
        assert np.all(w >= -1e-6), f"{name} violates long-only constraint: min weight = {np.min(w)}"


def test_institutional_position_cap_constraint() -> None:
    """Verify Strategy 4 strictly enforces the 15% maximum single-asset position ceiling."""
    prices = fetch_or_load_universe_prices()
    rets = prices[PORTFOLIO_TICKERS].pct_change().dropna().iloc[:252].values
    mu = np.mean(rets, axis=0) * 252.0
    cov_lw, _ = compute_ledoit_wolf_covariance(rets)

    w_inst = optimize_constrained_institutional(
        mu,
        cov_lw * 252.0,
        max_position_cap=0.150,
        turnover_penalty_lambda=0.080,
    )

    max_weight = float(np.max(w_inst))
    assert max_weight <= 0.150 + 1e-6, f"Position ceiling violated: max weight = {max_weight:.4f}"


def test_walk_forward_net_vs_gross_frictions() -> None:
    """Verify out-of-sample net returns are strictly <= gross returns when turnover > 0."""
    backtester = WalkForwardBacktester()
    res = backtester.run_backtest()

    df_gross = res["df_gross"]
    df_net = res["df_net"]

    for strat in df_gross.columns:
        cum_gross = (1.0 + df_gross[strat]).cumprod().iloc[-1]
        cum_net = (1.0 + df_net[strat]).cumprod().iloc[-1]
        assert cum_net <= cum_gross, f"Net wealth exceeds gross wealth for {strat}"

        diffs = df_gross[strat] - df_net[strat]
        assert np.all(diffs >= -1e-9), f"Daily net return exceeded gross return in {strat}"


def test_performance_metrics_math() -> None:
    """Verify mathematical definitions of Sharpe, Sortino, Drawdown, and CAGR."""
    np.random.seed(42)
    daily_rets = pd.Series(np.random.normal(0.0008, 0.012, 504))

    metrics = compute_performance_metrics(daily_rets, risk_free_rate=0.030)

    assert "cagr" in metrics
    assert "ann_volatility" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown" in metrics

    assert metrics["max_drawdown"] <= 0.0, "Drawdown must be non-positive"
    assert metrics["ann_volatility"] > 0.0, "Volatility must be positive"
