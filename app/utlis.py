import pandas as pd
import numpy as np
import yfinance as yf


# -------------------------------
# Constants
# -------------------------------
REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


# -------------------------------
# Helper
# -------------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns returned by yfinance (if any)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# -------------------------------
# Core Functions
# -------------------------------
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV data using yfinance.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False
    )

    if df is None or df.empty:
        raise ValueError(
            f"No data found for {ticker}. Check ticker or date range."
        )

    df = flatten_columns(df)
    df.index = pd.to_datetime(df.index)

    return df


def validate_df(df: pd.DataFrame) -> None:
    """
    Validate OHLCV DataFrame.
    """
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")

    if df.index.has_duplicates:
        raise ValueError("Duplicate index values found")

    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)


def add_features(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    vol_window: int
) -> pd.DataFrame:
    """
    Add features exactly as Phase-1 notebook:
    - daily_return
    - SMA_Short
    - SMA_Long
    - Rolling_Volatility (price-based, not annualized)
    """
    if df.empty:
        raise ValueError("Cannot compute features on empty DataFrame")

    DF = df.copy()

    # Daily return (same formula as notebook)
    DF["daily_return"] = (
        DF["Close"] - DF["Close"].shift(1)
    ) / DF["Close"].shift(1)

    # Simple Moving Averages
    DF["SMA_Short"] = DF["Close"].rolling(window=short_window).mean()
    DF["SMA_Long"] = DF["Close"].rolling(window=long_window).mean()

    # Rolling volatility (price-based, notebook-style)
    DF["Rolling_Volatility"] = (
        DF["Close"].rolling(window=vol_window).std()
    )

    return DF
