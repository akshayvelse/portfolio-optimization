"""Universe definition and historical market price caching manager.

Maintains a 12-ticker cross-sector US equity universe alongside SPY benchmark:
- Technology: AAPL, MSFT, NVDA
- Healthcare: JNJ, UNH
- Financials: JPM, GS
- Consumer Discretionary: AMZN
- Consumer Staples: PG
- Industrials: CAT
- Energy: XOM
- Communication Services: GOOGL
- Benchmark: SPY

Persists daily adjusted close price series and market capitalizations for offline execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PORTFOLIO_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "JNJ",
    "UNH",
    "JPM",
    "GS",
    "AMZN",
    "PG",
    "CAT",
    "XOM",
    "GOOGL",
]
BENCHMARK_TICKER = "SPY"
ALL_TICKERS = PORTFOLIO_TICKERS + [BENCHMARK_TICKER]


def ensure_data_directory() -> Path:
    """Ensure data caching directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def fetch_or_load_universe_prices(
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Retrieve and cache daily adjusted close prices for universe assets and benchmark."""
    ensure_data_directory()
    cache_path = DATA_DIR / "prices.csv"

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    raw_data = yf.download(
        ALL_TICKERS,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if "Adj Close" in raw_data.columns:
        prices_df = raw_data["Adj Close"].copy()
    else:
        prices_df = raw_data["Close"].copy()

    # Forward fill then backward fill any minor exchange holiday discrepancies
    prices_df = prices_df.ffill().bfill()
    prices_df = prices_df[ALL_TICKERS]  # Ensure column ordering

    prices_df.to_csv(cache_path)
    return prices_df


def fetch_or_load_universe_market_caps(
    force_refresh: bool = False,
) -> pd.Series:
    """Retrieve and cache latest market capitalizations for market-cap weighting."""
    ensure_data_directory()
    cache_path = DATA_DIR / "market_caps.csv"

    if cache_path.exists() and not force_refresh:
        s = pd.read_csv(cache_path, index_col=0).squeeze("columns")
        return s

    mcaps: dict[str, float] = {}
    for ticker in PORTFOLIO_TICKERS:
        t = yf.Ticker(ticker)
        mcap = float(t.info.get("marketCap") or 0.0)
        mcaps[ticker] = mcap

    s = pd.Series(mcaps, name="market_cap")
    s.to_csv(cache_path)
    return s


if __name__ == "__main__":
    prices = fetch_or_load_universe_prices(force_refresh=True)
    mcaps = fetch_or_load_universe_market_caps(force_refresh=True)
    print("Universe prices cached:", prices.shape)
    print("Universe market caps cached:", len(mcaps))
