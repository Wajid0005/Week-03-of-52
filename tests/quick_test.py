# tests/quick_test.py
from app.utils import fetch_data, validate_df, add_features
from app.plots import plot_price_ma, plot_return_dist
# NOTE: run in a normal Python env (not Streamlit). Plotly figures may open in browser if you call fig.show()

# 1) fetch (returns DF)
DF = fetch_data("AAPL", "2020-01-01", "2021-01-01")
validate_df(DF)

# 2) add features (returns DF_out)
DF_out = add_features(DF, short_window=5, long_window=20, vol_window=10)

# 3) quick print to confirm columns (uppercase names preserved)
print(DF_out.columns.tolist())

# 4) plot (will return Plotly figures)
fig_price = plot_price_ma(DF_out, show_sma=True)
fig_hist = plot_return_dist(DF_out)

# To view interactively (optional outside Streamlit):
# fig_price.show()
# fig_hist.show()
