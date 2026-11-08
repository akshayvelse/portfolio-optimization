# Portfolio Optimization & Walk-Forward Friction Engine

An institutional portfolio construction and backtesting system comparing 4 systematic allocation strategies against the S&P 500 (SPY) benchmark over 5+ years of out-of-sample daily data, incorporating Ledoit-Wolf shrinkage covariance, position caps, and realistic transaction cost deductions.

## Directory Architecture

```
03_portfolio_optimization/
├── README.md
├── backtest_tear_sheet.md
├── data/
│   ├── prices.csv
│   └── market_caps.csv
├── src/
│   ├── __init__.py
│   ├── universe.py
│   ├── optimizers.py
│   ├── backtester.py
│   └── tear_sheet_plots.py
└── output/
    ├── cumulative_returns_drawdown.png
    ├── weight_allocation_comparison.png
    └── performance_summary.csv
```

## Strategies Evaluated

1. **Equal-Weighted ($1/N$):** Naive balanced diversification across all 12 universe equities ($w_i = 1/12$).
2. **Market-Cap Weighted:** Value-weighted allocation proportional to aggregate enterprise equity capitalization.
3. **Unconstrained Mean-Variance Optimization:** Maximum Sharpe ratio optimization using unregularized sample covariance and trailing historical mean drift. Prone to corner solutions and parameter estimation noise.
4. **Constrained Institutional Allocation:** Maximum Sharpe ratio optimization combining:
   - Analytical Ledoit-Wolf shrinkage covariance with equicorrelation target.
   - Long-only constraint ($w_i \ge 0$).
   - Strict 15.0% single-position weight ceiling ($w_i \le 0.150$).
   - L1 turnover penalty in the optimization objective to penalize excessive rebalancing drag.
5. **Benchmark:** SPDR S&P 500 ETF Trust (SPY).

## Transaction Cost & Rebalancing Friction Model

- **Rebalance Frequency:** Quarterly (63 trading days) with a rolling 252-day lookback window.
- **Execution Cost Rate:** 15 basis points (0.0015) per dollar of turnover (10 bps broker commission + 5 bps market slippage).
- **Friction Drag:** Deducted directly from portfolio equity value on each quarterly rebalance date based on two-sided portfolio displacement $\sum_{i=1}^N |w_{i, t} - w_{i, t-1}^+|$.

## Execution

```bash
# Retrieve and cache universe daily prices and market caps
python 03_portfolio_optimization/src/universe.py

# Run unit optimization algorithms
python 03_portfolio_optimization/src/optimizers.py

# Execute walk-forward simulation and output performance summary
python 03_portfolio_optimization/src/backtester.py

# Render institutional tear sheet visual charts
python 03_portfolio_optimization/src/tear_sheet_plots.py
```
