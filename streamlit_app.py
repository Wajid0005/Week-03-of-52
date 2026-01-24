"""
Streamlit UI for Week03 - Interactive Financial Dashboard.
Uses Phase-1 feature logic from utils.py and Plotly visuals from plots.py.
"""

# -------------------------
# Imports
# -------------------------
import os
import sys
from datetime import date, datetime
import io

import streamlit as st
import pandas as pd
import numpy as np

# Ensure current directory is in Python path (robust for Streamlit Cloud)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils import fetch_data, validate_df, add_features
from plots import (
    plot_price_ma,
    plot_volume,
    plot_return_dist,
    plot_volatility
)

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Week03 — Interactive Financial Dashboard",
    layout="wide"
)

# -------------------------
# Cached fetch
# -------------------------
@st.cache_data(show_spinner=False)
def cached_fetch(ticker: str, start_iso: str, end_iso: str, refresh_token: int) -> pd.DataFrame:
    return fetch_data(ticker, start_iso, end_iso)

# -------------------------
# Session state
# -------------------------
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = 0

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Controls")

    TICKER = st.text_input("Ticker", value="AAPL").strip()
    START_DATE = st.date_input("Start date", value=date(2020, 1, 1))

    end_mode = st.selectbox("End date", ["Today", "Custom"])
    END_DATE = st.date_input("End date", value=date.today()) if end_mode == "Custom" else date.today()

    SHORT_SMA = st.slider("Short SMA", 2, 50, 10)
    LONG_SMA = st.slider("Long SMA", 5, 200, 50)
    VOL_WINDOW = st.slider("Volatility window", 5, 252, 20)

    SHOW_SMA = st.checkbox("Show SMA", True)
    SHOW_VOLUME = st.checkbox("Show Volume", True)
    SHOW_RETURNS = st.checkbox("Show Returns", True)
    SHOW_VOL = st.checkbox("Show Volatility", True)

    if st.button("Refresh"):
        st.session_state.refresh_token += 1
        st.rerun()

# -------------------------
# Validation
# -------------------------
if not TICKER:
    st.error("Please enter a valid ticker symbol.")
    st.stop()

if START_DATE >= END_DATE:
    st.error("Start date must be before end date.")
    st.stop()

# -------------------------
# Fetch data
# -------------------------
try:
    DF = cached_fetch(
        TICKER,
        START_DATE.isoformat(),
        END_DATE.isoformat(),
        st.session_state.refresh_token
    )
    validate_df(DF)
except Exception as e:
    st.error(str(e))
    st.stop()

# -------------------------
# Feature engineering
# -------------------------
DF_out = add_features(DF, SHORT_SMA, LONG_SMA, VOL_WINDOW)

if LONG_SMA <= SHORT_SMA:
    st.warning("Long SMA is not greater than Short SMA (allowed but uncommon).")

# -------------------------
# Header & stats
# -------------------------
st.title("📊 Interactive Financial Dashboard")

st.markdown(
    f"""
**Ticker:** `{TICKER.upper()}`  
**Date range:** {DF_out.index.min().date()} → {DF_out.index.max().date()}  
**Rows:** {len(DF_out)}
"""
)

mean_daily = DF_out["daily_return"].dropna().mean()
ann_vol = DF_out["daily_return"].dropna().std() * np.sqrt(252)

c1, c2, _ = st.columns(3)
c1.metric("Mean Daily Return", f"{mean_daily:.4%}")
c2.metric("Annualized Volatility", f"{ann_vol:.2%}")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Price", "Volume", "Returns", "Volatility", "Raw Data"]
)

with tab1:
    st.plotly_chart(plot_price_ma(DF_out, SHOW_SMA), use_container_width=True)

with tab2:
    if SHOW_VOLUME:
        st.plotly_chart(plot_volume(DF_out), use_container_width=True)

with tab3:
    if SHOW_RETURNS:
        st.plotly_chart(plot_return_dist(DF_out), use_container_width=True)

with tab4:
    if SHOW_VOL:
        st.plotly_chart(plot_volatility(DF_out), use_container_width=True)

with tab5:
    st.dataframe(DF_out.tail(500))
    csv = DF_out.to_csv(index=True).encode()
    st.download_button("Download CSV", csv, f"{TICKER}.csv", "text/csv")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Week 03 • Streamlit + Plotly • Stock-agnostic dashboard")
