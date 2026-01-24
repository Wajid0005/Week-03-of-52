# app/streamlit_app.py
"""
Streamlit UI for Week03 - Interactive Financial Dashboard (Phase 2).
- Uses DF naming and column names consistent with Phase-1 notebook.
- Relies on utils.py (fetch_data, validate_df, add_features) and plots.py (plot_* functions).
"""

import os
import sys
from datetime import date, datetime
import io

import streamlit as st
import pandas as pd
import numpy as np

# Ensure local `app` folder modules (utils, plots) are importable regardless of how streamlit run is invoked
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)
    
from utils import fetch_data, validate_df, add_features
from plots import plot_price_ma, plot_volume, plot_return_dist, plot_volatility



# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Week03 — Interactive Financial Dashboard", layout="wide")

# -------------------------
# Helper: cached fetch
# -------------------------
@st.cache_data(show_spinner=False)
def cached_fetch(ticker: str, start_iso: str, end_iso: str, _refresh_counter: int = 0) -> pd.DataFrame:
    """
    Cached wrapper around fetch_data. The _refresh_counter is an integer token that
    can be changed to force cache invalidation when user presses "Refresh".
    """
    DF = fetch_data(ticker, start_iso, end_iso)
    return DF


# -------------------------
# Session state for refresh
# -------------------------
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = 0

# -------------------------
# Sidebar: Inputs (exact widgets as required)
# -------------------------
with st.sidebar:
    st.header("Controls")

    TICKER = st.text_input("Ticker", value="AAPL").strip()
    START_DATE = st.date_input("Start date", value=date(2020, 1, 1))

    end_mode = st.selectbox("End date", options=["Today", "Custom"], index=0)
    if end_mode == "Custom":
        END_DATE = st.date_input("End date", value=date.today())
    else:
        END_DATE = date.today()

    SHORT_SMA = st.slider("Short SMA", min_value=2, max_value=50, value=10)
    LONG_SMA = st.slider("Long SMA", min_value=5, max_value=200, value=50)
    VOL_WINDOW = st.slider("Volatility window", min_value=5, max_value=252, value=20)

    SHOW_SMA = st.checkbox("Show SMA", value=True)
    SHOW_VOLUME = st.checkbox("Show Volume", value=True)
    SHOW_RETURNS = st.checkbox("Show Returns", value=True)
    SHOW_VOL = st.checkbox("Show Volatility", value=True)

    # Refresh button increments token to bust cache for next fetch
    if st.button("Refresh"):
        st.session_state.refresh_token += 1
        st.experimental_rerun()

# -------------------------
# Input validations (pre-fetch)
# -------------------------
# Basic ticker check
if not TICKER:
    st.error("Please enter a ticker symbol (e.g., AAPL or RELIANCE.NS).")
    st.stop()

# Date sanity
if isinstance(START_DATE, datetime):
    START_DATE = START_DATE.date()
if isinstance(END_DATE, datetime):
    END_DATE = END_DATE.date()

if START_DATE >= END_DATE:
    st.error("Start date must be earlier than end date.")
    st.stop()

# -------------------------
# Fetch data (cached)
# -------------------------
fetch_error = None
try:
    # yfinance expects ISO strings
    DF = cached_fetch(TICKER, START_DATE.isoformat(), END_DATE.isoformat(), st.session_state.refresh_token)
    validate_df(DF)
except Exception as e:
    fetch_error = str(e)
    DF = None

if fetch_error:
    st.error(fetch_error)
    st.stop()

# -------------------------
# Feature engineering
# -------------------------
# Warn if windows exceed available data length
n_rows = len(DF)
max_window = max(SHORT_SMA, LONG_SMA, VOL_WINDOW)
if n_rows < max_window:
    st.warning(
        f"Data contains {n_rows} rows but the largest window is {max_window}. "
        "Some indicator columns will be NaN until enough data is available."
    )

if LONG_SMA <= SHORT_SMA:
    st.warning("Long SMA is not greater than Short SMA. This is allowed but uncommon.")

# Compute features (returns a new DF_out)
try:
    DF_out = add_features(DF, short_window=SHORT_SMA, long_window=LONG_SMA, vol_window=VOL_WINDOW)
except Exception as e:
    st.error(f"Error while computing features: {e}")
    st.stop()

# -------------------------
# Header & summary
# -------------------------
st.title("📊 Interactive Financial Dashboard — Phase 2")
st.markdown(
    f"**Ticker:** `{TICKER.upper()}` &nbsp;&nbsp; • &nbsp;&nbsp;"
    f"**Date range:** {DF_out.index.min().date()} → {DF_out.index.max().date()} &nbsp;&nbsp; • &nbsp;&nbsp;"
    f"**Rows:** {n_rows}"
)

# Compute quick stats (guard against NaNs)
mean_daily = DF_out["daily_return"].dropna().mean() if "daily_return" in DF_out.columns else np.nan
ann_vol = DF_out["daily_return"].dropna().std() * np.sqrt(252) if "daily_return" in DF_out.columns else np.nan

col1, col2, col3 = st.columns(3)
col1.metric("Mean daily return (mean)", f"{mean_daily:.4%}" if pd.notna(mean_daily) else "n/a")
col2.metric("Annualized vol (approx)", f"{ann_vol:.2%}" if pd.notna(ann_vol) else "n/a")
col3.write("")  # placeholder for layout symmetry

# -------------------------
# Tabs for visuals
# -------------------------
tab_price, tab_volume, tab_returns, tab_vol, tab_raw = st.tabs(
    ["Price & Indicators", "Volume", "Returns", "Volatility", "Raw Data"]
)

with tab_price:
    st.subheader("Price chart")
    try:
        fig_price = plot_price_ma(DF_out, show_sma=SHOW_SMA)
        st.plotly_chart(fig_price, use_container_width=True)
    except Exception as e:
        st.error(f"Could not render price chart: {e}")

    # Optionally show volume as secondary or separate below (user chose Show Volume)
    if SHOW_VOLUME:
        try:
            fig_vol = plot_volume(DF_out)
            st.plotly_chart(fig_vol, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render volume chart: {e}")

with tab_volume:
    st.subheader("Volume")
    if SHOW_VOLUME:
        try:
            st.plotly_chart(plot_volume(DF_out), use_container_width=True)
        except Exception as e:
            st.error(f"Could not render volume chart: {e}")
    else:
        st.info("Volume display is turned off. Toggle 'Show Volume' in the sidebar to enable.")

with tab_returns:
    st.subheader("Returns distribution")
    if SHOW_RETURNS:
        try:
            st.plotly_chart(plot_return_dist(DF_out), use_container_width=True)
        except Exception as e:
            st.error(f"Could not render returns histogram: {e}")
    else:
        st.info("Returns display is turned off. Toggle 'Show Returns' in the sidebar to enable.")

with tab_vol:
    st.subheader("Rolling volatility")
    if SHOW_VOL:
        try:
            st.plotly_chart(plot_volatility(DF_out), use_container_width=True)
        except Exception as e:
            st.error(f"Could not render volatility chart: {e}")
    else:
        st.info("Volatility display is turned off. Toggle 'Show Volatility' in the sidebar to enable.")

with tab_raw:
    st.subheader("Raw data & download")
    st.write("Preview (last 500 rows):")
    st.dataframe(DF_out.tail(500))

    # CSV download
    csv_buffer = io.StringIO()
    DF_out.to_csv(csv_buffer, index=True)
    csv_bytes = csv_buffer.getvalue().encode()
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"{TICKER.upper()}_{START_DATE.isoformat()}_{END_DATE.isoformat()}.csv",
        mime="text/csv"
    )

# -------------------------
# Collapsible: Feature summary & code notes
# -------------------------
with st.expander("Feature summary & notes (Phase 1 logic)"):
    st.markdown(
        """
- `daily_return` : simple percent change on Close (same as notebook).
- `SMA_Short` / `SMA_Long` : rolling mean on Close.
- `Rolling_Volatility` : rolling std of Close (price-based), *not annualized* (keeps Phase-1 behavior).
- Caching: `@st.cache_data` is used to cache yfinance downloads by (ticker, start, end, refresh_token).
- If you change indicator windows or refresh, charts update accordingly.
"""
    )

# -------------------------
# Footer: quick troubleshooting
# -------------------------
st.markdown("---")
st.caption("If a ticker returns no data, check ticker format (e.g., use RELIANCE.NS for NSE) and verify date ranges. "
           "For large date ranges, caching reduces repeated downloads.")
