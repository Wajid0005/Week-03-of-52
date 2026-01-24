# app/utils.py
import pandas as pd
import numpy as np
import yfinance as yf

# Keep the notebook naming and behavior: DF is the DataFrame variable name.
REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


def flatten_columns(DF: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns from yfinance if present.
    Returns the same DF object type with flattened columns.
    """
    if isinstance(DF.columns, pd.MultiIndex):
        DF.columns = DF.columns.get_level_values(0)
    return DF


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV using yfinance and return DF (DatetimeIndex).
    Mirrors notebook behavior: returns DF or raises ValueError if empty.
    """
    DF = yf.download(ticker, start=start, end=end, progress=False)
    if DF is None or DF.empty:
        raise ValueError(f"No data found for {ticker}. Check ticker or date range.")
    DF = flatten_columns(DF)
    DF.index = pd.to_datetime(DF.index)
    return DF


def validate_df(DF: pd.DataFrame) -> None:
    """
    Validate the DF structure (same checks as Phase-1 expectations).
    Raises ValueError for problem cases.
    """
    missing = REQUIRED_COLS - set(DF.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not isinstance(DF.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")

    if DF.index.has_duplicates:
        raise ValueError("Duplicate timestamps detected in DF.index")

    # Notebook expects sorted index; enforce it
    if not DF.index.is_monotonic_increasing:
        DF.sort_index(inplace=True)


def add_features(DF: pd.DataFrame, short_window: int, long_window: int, vol_window: int) -> pd.DataFrame:
    """
    Add Phase-1 style features to DF and return a new DataFrame (copy).
    - daily_return (same formula as notebook)
    - SMA_Short
    - SMA_Long
    - Rolling_Volatility (price-based rolling std, NOT annualized) 
    """
    if DF is None or DF.empty:
        raise ValueError("Cannot compute features on an empty DF")

    # Work on a copy so we don't unexpectedly mutate notebook variable if user reuses it
    DF_out = DF.copy()

    # daily_return: same as your notebook (simple pct change formula)
    DF_out["daily_return"] = (DF_out["Close"] - DF_out["Close"].shift(1)) / DF_out["Close"].shift(1)

    # SMAs on Close (keep names as in notebook)
    DF_out["SMA_Short"] = DF_out["Close"].rolling(window=short_window).mean()
    DF_out["SMA_Long"] = DF_out["Close"].rolling(window=long_window).mean()

    # Rolling volatility (price-based as in the notebook)
    DF_out["Rolling_Volatility"] = DF_out["Close"].rolling(window=vol_window).std()

    return DF_out

