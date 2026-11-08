# QUANTITATIVE PORTFOLIO OPTIMIZATION & BACKTEST TEAR SHEET

**UNIVERSE:** 12 Large-Cap US Equities (AAPL, MSFT, NVDA, JNJ, UNH, JPM, GS, AMZN, PG, CAT, XOM, GOOGL)  
**BENCHMARK:** SPDR S&P 500 ETF Trust (SPY)  
**TEST HORIZON:** 2020-01-02 through 2024-12-30 (1,257 Out-of-Sample Trading Days)  
**ESTIMATION WINDOW:** Rolling 252-day lookback, quarterly rebalancing (63 trading days)  
**TRANSACTION FRICTIONS:** 15 bps per turn (10 bps commission + 5 bps slippage)  
**RISK-FREE PROXY:** 3.00% annual benchmark  
**DATE:** November 8, 2026  

---

## 1. EMPIRICAL PERFORMANCE & RISK SUMMARY TABLE

The walk-forward out-of-sample backtest evaluates gross returns alongside net-of-friction returns:

| Strategy | CAGR | Annual Volatility | Sharpe Ratio | Sortino Ratio | Maximum Drawdown | Calmar Ratio | Annual Turnover | Rebalancing Friction Drag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Equal-Weighted (1/N) [Gross]** | 26.22% | 22.12% | 1.05 | 1.50 | -31.40% | 0.83 | 38.8% | — |
| **1. Equal-Weighted (1/N) [Net]** | 26.14% | 22.12% | 1.05 | 1.50 | -31.40% | 0.83 | 38.8% | **-7.3 bps** |
| **2. Market-Cap Weighted [Gross]** | 36.75% | 28.77% | 1.17 | 1.70 | -34.66% | 1.06 | 55.2% | — |
| **2. Market-Cap Weighted [Net]** | 36.64% | 28.77% | 1.17 | 1.69 | -34.70% | 1.06 | 55.2% | **-11.4 bps** |
| **3. Unconstrained MVO [Gross]** | 27.80% | 28.13% | 0.88 | 1.30 | -29.43% | 0.94 | 369.3% | — |
| **3. Unconstrained MVO [Net]** | 27.09% | 28.13% | 0.86 | 1.26 | -29.43% | 0.92 | 369.3% | **-70.7 bps** |
| **4. Constrained Institutional [Gross]** | 23.21% | 23.35% | 0.87 | 1.26 | -31.43% | 0.74 | 190.9% | — |
| **4. Constrained Institutional [Net]** | 22.86% | 23.35% | 0.85 | 1.24 | -31.43% | 0.73 | 190.9% | **-35.2 bps** |
| **Benchmark: S&P 500 (SPY)** | 14.37% | 21.00% | 0.54 | 0.75 | -33.72% | 0.43 | 0.0% | **0.0 bps** |

---

## 2. QUANTITATIVE ANALYSIS: FAILURE MODES OF UNCONSTRAINED MVO

### The Markowitz "Error Maximizer"
Modern Portfolio Theory assumes the first two moments of asset returns ($\mu$ and $\Sigma$) are known with certainty. In practical out-of-sample execution, historical sample means are poor estimators of forward drift, while sample covariance matrices invert estimation noise into extreme asset weightings:
1. **Corner Solutions:** Across the 19 quarterly rebalancing dates, Strategy 3 allocated 0.0% weight to an average of 6.8 out of the 12 available universe assets. The unconstrained optimizer concentrated up to 37.3% in single high-momentum assets (e.g., NVDA, AAPL) during bull expansions, followed by abrupt 100% liquidations when historical trailing returns Mean-Reverted.
2. **Turnover Drag & Capital Erosion:** Unconstrained MVO churned 369.3% annualized turnover. Applying standard institutional execution friction (15 bps combined fee and market impact) destroyed **70.7 basis points** of annual return. Over the 5-year testing horizon, this friction drag accumulated to a 3.54% terminal wealth reduction solely attributable to unconstrained rebalancing churn.

### Regularization via Ledoit-Wolf Shrinkage & Hard Constraints
Strategy 4 resolves these failure modes through three systematic risk controls:
1. **Covariance Conditioning:** Shrinks the sample covariance matrix toward an equicorrelation target via optimal intensity $\delta^* \in [0, 1]$. This dampens spurious off-diagonal sample correlation spikes and conditions the Hessian matrix.
2. **Position Ceiling (15.0% Cap):** Eliminates idiosyncratic concentration risk, preventing single-stock earnings surprises from dominating aggregate portfolio variance.
3. **Turnover Penalty ($\lambda = 0.08$):** Penalizes portfolio displacement relative to drifted holdings $w_{t-1}^+$, halving annualized turnover from 369.3% down to 190.9% and cutting friction drag by 50.2% (from 70.7 bps to 35.2 bps).

---

## 3. ASSET ALLOCATION DYNAMICS

Comparative allocation inspection confirms structural stability differences:
- **Unconstrained MVO (Chart Top):** Highly fragmented weight profiles where individual asset allocations swing by 25-35% within a single quarter, reflecting parameter sensitivity to trailing 252-day return outliers.
- **Constrained Institutional (Chart Bottom):** Continuous diversification across all 12 assets with dynamic weight clipping along the 15.0% line, maintaining equity factor diversification across defensive consumer staples (PG, JNJ) and cyclical growth (MSFT, AMZN, NVDA).
