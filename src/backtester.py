"""Walk-forward portfolio backtesting engine with realistic transaction frictions.

Simulates quarterly rebalanced allocations across 12 cross-sector equities:
- Rolling 252-day lookback estimation window
- 15 bps total execution frictions (10 bps broker commission + 5 bps market slippage)
- Evaluates gross and net-of-friction risk-adjusted return metrics:
  CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Calmar Ratio, Annualized Turnover
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from .optimizers import (
        compute_ledoit_wolf_covariance,
        optimize_constrained_institutional,
        optimize_equal_weight,
        optimize_market_cap_weight,
        optimize_unconstrained_mvo,
    )
    from .universe import (
        BENCHMARK_TICKER,
        PORTFOLIO_TICKERS,
        fetch_or_load_universe_market_caps,
        fetch_or_load_universe_prices,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from optimizers import (
        compute_ledoit_wolf_covariance,
        optimize_constrained_institutional,
        optimize_equal_weight,
        optimize_market_cap_weight,
        optimize_unconstrained_mvo,
    )
    from universe import (
        BENCHMARK_TICKER,
        PORTFOLIO_TICKERS,
        fetch_or_load_universe_market_caps,
        fetch_or_load_universe_prices,
    )

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

LOOKBACK_WINDOW = 252  # 1 trading year
REBALANCE_FREQ = 63  # Quarterly (~63 trading days)
COMMISSION_BPS = 0.0010  # 10 bps
SLIPPAGE_BPS = 0.0005  # 5 bps
TOTAL_FRICTION_RATE = COMMISSION_BPS + SLIPPAGE_BPS  # 15 bps (0.0015)
RISK_FREE_RATE = 0.030  # 3.0% annual risk-free proxy


def compute_performance_metrics(
    returns_series: pd.Series,
    annual_turnover: float = 0.0,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, float]:
    """Calculate institutional risk and return attribution metrics."""
    clean_rets = returns_series.dropna()
    n_days = len(clean_rets)
    if n_days == 0:
        return {}

    # Cumulative growth
    cum_growth = (1.0 + clean_rets).cumprod()
    total_return = float(cum_growth.iloc[-1]) - 1.0

    # Compound Annual Growth Rate (CAGR)
    cagr = float(cum_growth.iloc[-1] ** (252.0 / float(n_days))) - 1.0

    # Annualized Volatility
    ann_vol = float(clean_rets.std() * np.sqrt(252.0))

    # Sharpe Ratio
    sharpe = (cagr - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    # Sortino Ratio (downside deviation relative to daily risk-free rate)
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    downside_diffs = np.minimum(clean_rets.values - daily_rf, 0.0)
    downside_vol = float(np.sqrt(np.mean(downside_diffs**2)) * np.sqrt(252.0))
    sortino = (cagr - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0

    # Maximum Drawdown
    peaks = cum_growth.cummax()
    drawdowns = (cum_growth - peaks) / peaks
    max_drawdown = float(drawdowns.min())

    # Calmar Ratio
    calmar = cagr / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0

    return {
        "cagr": cagr,
        "ann_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "annualized_turnover": annual_turnover,
        "cumulative_return": total_return,
    }


class WalkForwardBacktester:
    """Executes rolling quarterly portfolio rebalancing and tracks friction drag."""

    def __init__(
        self,
        prices_df: pd.DataFrame | None = None,
        market_caps: pd.Series | None = None,
    ) -> None:
        if prices_df is None:
            prices_df = fetch_or_load_universe_prices()
        if market_caps is None:
            market_caps = fetch_or_load_universe_market_caps()

        self.prices_df = prices_df
        self.market_caps = market_caps
        self.asset_tickers = PORTFOLIO_TICKERS
        self.asset_returns = prices_df[self.asset_tickers].pct_change().dropna()
        self.benchmark_returns = (
            prices_df[BENCHMARK_TICKER].pct_change().dropna()
        )

    def run_backtest(self) -> dict[str, Any]:
        """Execute walk-forward out-of-sample simulation across all 4 strategies."""
        dates = self.asset_returns.index
        n_total_days = len(dates)

        # Simulation indices start after lookback window
        start_idx = LOOKBACK_WINDOW
        rebalance_indices = list(range(start_idx, n_total_days, REBALANCE_FREQ))

        # Containers for daily returns and rebalance records
        strategies = ["equal_weight", "market_cap", "unconstrained_mvo", "constrained_inst"]

        daily_returns_gross: dict[str, list[float]] = {s: [] for s in strategies}
        daily_returns_net: dict[str, list[float]] = {s: [] for s in strategies}
        weight_history: dict[str, list[np.ndarray]] = {s: [] for s in strategies}
        turnover_history: dict[str, list[float]] = {s: [] for s in strategies}
        rebalance_dates: list[pd.Timestamp] = []

        # Current portfolio holdings weights (drifted daily)
        current_weights = {
            s: np.ones(len(self.asset_tickers)) / float(len(self.asset_tickers))
            for s in strategies
        }

        # Market-cap weights are fixed baseline
        mcap_weights = optimize_market_cap_weight(self.market_caps)

        # Iterate day-by-day through the out-of-sample test horizon
        for day_idx in range(start_idx, n_total_days):
            current_date = dates[day_idx]
            is_rebalance_day = day_idx in rebalance_indices

            # Today's individual asset returns
            day_asset_rets = self.asset_returns.iloc[day_idx].values

            if is_rebalance_day:
                rebalance_dates.append(current_date)
                # Historical lookback estimation window [day_idx - LOOKBACK_WINDOW : day_idx]
                lookback_rets = self.asset_returns.iloc[day_idx - LOOKBACK_WINDOW : day_idx].values
                mu = np.mean(lookback_rets, axis=0) * 252.0
                cov_sample = np.cov(lookback_rets, rowvar=False) * 252.0
                cov_lw, _ = compute_ledoit_wolf_covariance(lookback_rets)
                cov_lw_annual = cov_lw * 252.0

                # Compute target allocation weights for each strategy
                target_weights = {
                    "equal_weight": optimize_equal_weight(len(self.asset_tickers)),
                    "market_cap": mcap_weights.copy(),
                    "unconstrained_mvo": optimize_unconstrained_mvo(mu, cov_sample, RISK_FREE_RATE),
                    "constrained_inst": optimize_constrained_institutional(
                        mu,
                        cov_lw_annual,
                        previous_weights=current_weights["constrained_inst"],
                        risk_free_rate=RISK_FREE_RATE,
                        max_position_cap=0.150,
                        turnover_penalty_lambda=0.080,
                    ),
                }

                # Calculate turnover and deduction from rebalancing frictions
                for s in strategies:
                    new_w = target_weights[s]
                    old_w = current_weights[s]
                    turnover = float(np.sum(np.abs(new_w - old_w)))
                    friction_cost = turnover * TOTAL_FRICTION_RATE

                    turnover_history[s].append(turnover)
                    weight_history[s].append(new_w)
                    current_weights[s] = new_w

                    # Rebalance day return: gross return on new weights minus rebalancing cost
                    gross_ret = float(np.dot(new_w, day_asset_rets))
                    net_ret = gross_ret - friction_cost

                    daily_returns_gross[s].append(gross_ret)
                    daily_returns_net[s].append(net_ret)

                    # Update drifted weights at close of trading day
                    current_weights[s] = (new_w * (1.0 + day_asset_rets)) / (1.0 + gross_ret)

            else:
                # Non-rebalance day: passive drift
                for s in strategies:
                    w = current_weights[s]
                    gross_ret = float(np.dot(w, day_asset_rets))
                    net_ret = gross_ret  # No transaction cost incurred

                    daily_returns_gross[s].append(gross_ret)
                    daily_returns_net[s].append(net_ret)

                    # Update drifted weights
                    current_weights[s] = (w * (1.0 + day_asset_rets)) / (1.0 + gross_ret)

        out_of_sample_dates = dates[start_idx:]
        n_years = len(out_of_sample_dates) / 252.0

        # Construct Series objects
        df_gross = pd.DataFrame(daily_returns_gross, index=out_of_sample_dates)
        df_net = pd.DataFrame(daily_returns_net, index=out_of_sample_dates)
        benchmark_series = self.benchmark_returns.loc[out_of_sample_dates]

        # Compute summary performance table
        summary_rows: list[dict[str, Any]] = []

        for s in strategies:
            total_to = float(np.sum(turnover_history[s]))
            ann_to = (total_to / n_years) if n_years > 0 else 0.0

            m_gross = compute_performance_metrics(df_gross[s], annual_turnover=ann_to)
            m_net = compute_performance_metrics(df_net[s], annual_turnover=ann_to)

            strat_label = {
                "equal_weight": "1. Equal-Weighted (1/N)",
                "market_cap": "2. Market-Cap Weighted",
                "unconstrained_mvo": "3. Unconstrained MVO (Sample Cov)",
                "constrained_inst": "4. Constrained Institutional (LW + 15% Cap)",
            }[s]

            summary_rows.append(
                {
                    "Strategy": f"{strat_label} [Gross]",
                    "CAGR": m_gross["cagr"],
                    "Ann Vol": m_gross["ann_volatility"],
                    "Sharpe": m_gross["sharpe_ratio"],
                    "Sortino": m_gross["sortino_ratio"],
                    "Max DD": m_gross["max_drawdown"],
                    "Calmar": m_gross["calmar_ratio"],
                    "Ann Turnover": m_gross["annualized_turnover"],
                }
            )
            summary_rows.append(
                {
                    "Strategy": f"{strat_label} [Net]",
                    "CAGR": m_net["cagr"],
                    "Ann Vol": m_net["ann_volatility"],
                    "Sharpe": m_net["sharpe_ratio"],
                    "Sortino": m_net["sortino_ratio"],
                    "Max DD": m_net["max_drawdown"],
                    "Calmar": m_net["calmar_ratio"],
                    "Ann Turnover": m_net["annualized_turnover"],
                }
            )

        # Benchmark SPY metrics
        m_bench = compute_performance_metrics(benchmark_series, annual_turnover=0.0)
        summary_rows.append(
            {
                "Strategy": "Benchmark: S&P 500 (SPY)",
                "CAGR": m_bench["cagr"],
                "Ann Vol": m_bench["ann_volatility"],
                "Sharpe": m_bench["sharpe_ratio"],
                "Sortino": m_bench["sortino_ratio"],
                "Max DD": m_bench["max_drawdown"],
                "Calmar": m_bench["calmar_ratio"],
                "Ann Turnover": 0.0,
            }
        )

        perf_summary_df = pd.DataFrame(summary_rows)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        perf_summary_df.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)

        return {
            "df_gross": df_gross,
            "df_net": df_net,
            "benchmark": benchmark_series,
            "weight_history": weight_history,
            "turnover_history": turnover_history,
            "rebalance_dates": rebalance_dates,
            "performance_summary": perf_summary_df,
            "asset_tickers": self.asset_tickers,
        }


if __name__ == "__main__":
    backtester = WalkForwardBacktester()
    res = backtester.run_backtest()
    summary = res["performance_summary"]
    print("=== BACKTEST PERFORMANCE ATTRIBUTION SUMMARY ===")
    print(summary.to_string(index=False))
