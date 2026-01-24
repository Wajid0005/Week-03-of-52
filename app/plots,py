# app/plots.py
import plotly.graph_objects as go
import pandas as pd


def plot_price_ma(DF: pd.DataFrame, show_sma: bool = True) -> go.Figure:
    """
    Plot Close price and optional SMAs. Accepts DF (uppercase) exactly like notebook.
    """
    if DF is None or DF.empty:
        raise ValueError("DF is empty. Cannot plot price.")

    fig = go.Figure()

    # Close price
    fig.add_trace(
        go.Scatter(x=DF.index, y=DF["Close"], mode="lines", name="Close", line=dict(width=2))
    )

    # SMAs if present and requested
    if show_sma:
        if "SMA_Short" in DF.columns:
            fig.add_trace(go.Scatter(x=DF.index, y=DF["SMA_Short"], mode="lines", name="SMA_Short"))
        if "SMA_Long" in DF.columns:
            fig.add_trace(go.Scatter(x=DF.index, y=DF["SMA_Long"], mode="lines", name="SMA_Long"))

    fig.update_layout(
        title="Close Price & Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend_title_text="Series"
    )
    return fig


def plot_volume(DF: pd.DataFrame) -> go.Figure:
    """
    Bar chart of Volume. Uses DF exactly as in your notebook.
    """
    if DF is None or DF.empty:
        raise ValueError("DF is empty. Cannot plot volume.")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=DF.index, y=DF["Volume"], name="Volume"))
    fig.update_layout(title="Trading Volume", xaxis_title="Date", yaxis_title="Volume", hovermode="x")
    return fig


def plot_return_dist(DF: pd.DataFrame, nbins: int = 100) -> go.Figure:
    """
    Histogram of daily_return. Uses column name 'daily_return' per notebook.
    """
    if "daily_return" not in DF.columns:
        raise ValueError("DF does not contain 'daily_return'. Run add_features first.")

    returns = DF["daily_return"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=nbins, name="daily_return"))
    fig.update_layout(title="Daily Return Distribution", xaxis_title="daily_return", yaxis_title="Count")
    return fig


def plot_volatility(DF: pd.DataFrame) -> go.Figure:
    """
    Plot Rolling_Volatility (price-based, notebook-style).
    """
    if "Rolling_Volatility" not in DF.columns:
        raise ValueError("DF does not contain 'Rolling_Volatility'. Run add_features first.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DF.index, y=DF["Rolling_Volatility"], mode="lines", name="Rolling_Volatility"))
    fig.update_layout(title="Rolling Volatility (price-based)", xaxis_title="Date", yaxis_title="Volatility")
    return fig
