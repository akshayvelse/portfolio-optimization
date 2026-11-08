"""Institutional tear sheet visualization generator for portfolio backtest results.

Produces two high-resolution charts:
1. cumulative_returns_drawdown.png:
   - Top panel: Growth of $10,000 across four strategies (Gross & Net) and SPY benchmark
   - Bottom panel: Underwater drawdown profiles over the out-of-sample test horizon
2. weight_allocation_comparison.png:
   - Comparative stacked weight allocation dynamics: Unconstrained MVO vs Constrained Institutional
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .backtester import WalkForwardBacktester
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from backtester import WalkForwardBacktester

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

COLOR_EQUAL = "#2E75B6"  # Blue
COLOR_MCAP = "#70AD47"  # Green
COLOR_MVO = "#ED7D31"  # Orange
COLOR_INST = "#1B365D"  # Institutional Navy
COLOR_BENCH = "#7F7F7F"  # Slate Gray


def plot_cumulative_returns_and_drawdown(
    backtest_results: dict[str, Any], output_path: Path
) -> None:
    """Render top/bottom panel comparing equity curves and underwater drawdowns."""
    df_net = backtest_results["df_net"]
    benchmark = backtest_results["benchmark"]

    initial_capital = 10000.0

    # Calculate equity growth of $10,000
    equity_curves = (1.0 + df_net).cumprod() * initial_capital
    bench_curve = (1.0 + benchmark).cumprod() * initial_capital

    # Calculate drawdowns
    drawdowns = pd.DataFrame(index=df_net.index)
    for col in df_net.columns:
        curve = equity_curves[col]
        peak = curve.cummax()
        drawdowns[col] = (curve - peak) / peak

    bench_peak = bench_curve.cummax()
    bench_dd = (bench_curve - bench_peak) / bench_peak

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 9),
        dpi=300,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.25},
    )

    # Top Panel: Cumulative Growth
    ax_top.plot(
        equity_curves.index,
        equity_curves["constrained_inst"],
        label="Constrained Institutional (LW + 15% Cap) [Net]",
        color=COLOR_INST,
        linewidth=2.0,
    )
    ax_top.plot(
        equity_curves.index,
        equity_curves["unconstrained_mvo"],
        label="Unconstrained MVO (Sample Cov) [Net]",
        color=COLOR_MVO,
        linewidth=1.6,
        linestyle="--",
    )
    ax_top.plot(
        equity_curves.index,
        equity_curves["market_cap"],
        label="Market-Cap Weighted [Net]",
        color=COLOR_MCAP,
        linewidth=1.4,
    )
    ax_top.plot(
        equity_curves.index,
        equity_curves["equal_weight"],
        label="Equal-Weighted (1/N) [Net]",
        color=COLOR_EQUAL,
        linewidth=1.4,
    )
    ax_top.plot(
        bench_curve.index,
        bench_curve,
        label="Benchmark: S&P 500 (SPY)",
        color=COLOR_BENCH,
        linewidth=1.8,
        linestyle=":",
    )

    ax_top.set_title(
        "Out-of-Sample Portfolio Performance: Cumulative Growth of $10,000 (Net of 15 bps Frictions)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_top.set_ylabel("Portfolio Value (USD)", fontsize=10, fontweight="bold")
    ax_top.grid(True, linestyle="--", alpha=0.4)
    ax_top.legend(loc="upper left", frameon=True, fontsize=9)
    ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Bottom Panel: Underwater Drawdown
    ax_bot.plot(
        drawdowns.index,
        drawdowns["constrained_inst"] * 100.0,
        label="Constrained Institutional",
        color=COLOR_INST,
        linewidth=1.5,
    )
    ax_bot.plot(
        drawdowns.index,
        drawdowns["unconstrained_mvo"] * 100.0,
        label="Unconstrained MVO",
        color=COLOR_MVO,
        linewidth=1.2,
        linestyle="--",
    )
    ax_bot.plot(
        bench_dd.index,
        bench_dd * 100.0,
        label="S&P 500 (SPY)",
        color=COLOR_BENCH,
        linewidth=1.2,
        linestyle=":",
    )

    ax_bot.fill_between(
        drawdowns.index,
        drawdowns["constrained_inst"] * 100.0,
        0,
        alpha=0.15,
        color=COLOR_INST,
    )
    ax_bot.set_title("Underwater Portfolio Drawdown Profile (%)", fontsize=11, fontweight="bold")
    ax_bot.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
    ax_bot.set_xlabel("Out-of-Sample Rebalance Timeline", fontsize=10, fontweight="bold")
    ax_bot.grid(True, linestyle="--", alpha=0.4)
    ax_bot.set_ylim(-45.0, 2.0)
    ax_bot.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}%"))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_weight_allocation_comparison(
    backtest_results: dict[str, Any], output_path: Path
) -> None:
    """Render stacked area charts comparing Unconstrained MVO vs Constrained Institutional."""
    rebalance_dates = backtest_results["rebalance_dates"]
    weight_history = backtest_results["weight_history"]
    asset_tickers = backtest_results["asset_tickers"]

    mvo_weights = np.array(weight_history["unconstrained_mvo"])  # shape: (n_rebalances, 12)
    inst_weights = np.array(weight_history["constrained_inst"])

    fig, (ax_left, ax_right) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 10),
        dpi=300,
        sharex=True,
    )

    palette = plt.cm.tab20(np.linspace(0, 1, len(asset_tickers)))

    # Top Subplot: Strategy 3 (Unconstrained MVO)
    ax_left.stackplot(
        rebalance_dates,
        mvo_weights.T * 100.0,
        labels=asset_tickers,
        colors=palette,
        alpha=0.85,
    )
    ax_left.set_title(
        "Strategy 3: Unconstrained Mean-Variance Optimization (Sample Covariance)\nProne to Markowitz Error Maximization, Volatile Corner Solutions, and High Turnover",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax_left.set_ylabel("Allocation Weight (%)", fontsize=10, fontweight="bold")
    ax_left.set_ylim(0, 100)
    ax_left.grid(True, linestyle="--", alpha=0.3)
    ax_left.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, ncol=1)

    # Bottom Subplot: Strategy 4 (Constrained Institutional)
    ax_right.stackplot(
        rebalance_dates,
        inst_weights.T * 100.0,
        labels=asset_tickers,
        colors=palette,
        alpha=0.85,
    )
    ax_right.axhline(15.0, color="red", linestyle="--", linewidth=1.5, label="15% Maximum Position Ceiling")
    ax_right.set_title(
        "Strategy 4: Constrained Institutional Allocation (Ledoit-Wolf Shrinkage Covariance)\nStrict 15% Maximum Single-Position Cap and Turnover Regularization",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax_right.set_ylabel("Allocation Weight (%)", fontsize=10, fontweight="bold")
    ax_right.set_xlabel("Quarterly Rebalance Horizon", fontsize=10, fontweight="bold")
    ax_right.set_ylim(0, 100)
    ax_right.grid(True, linestyle="--", alpha=0.3)
    ax_right.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, ncol=1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_all_tear_sheet_artifacts() -> None:
    """Execute backtest, save performance summary table, and render all charts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backtester = WalkForwardBacktester()
    res = backtester.run_backtest()

    chart1_path = OUTPUT_DIR / "cumulative_returns_drawdown.png"
    chart2_path = OUTPUT_DIR / "weight_allocation_comparison.png"

    plot_cumulative_returns_and_drawdown(res, chart1_path)
    plot_weight_allocation_comparison(res, chart2_path)

    print(f"Generated cumulative returns chart: {chart1_path}")
    print(f"Generated weight allocation chart: {chart2_path}")


if __name__ == "__main__":
    generate_all_tear_sheet_artifacts()
