#!/usr/bin/env python3
import os, glob, json, pathlib
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Config (matches your existing notebooks) ----------
MODELS_DIR = os.getenv("MODELS_DIR", "models")   # your stock_mlops.ipynb writes models/<TICKER>.pkl
STOCK_DIR  = os.getenv("STOCK_DIR",  "Stock")    # Pull_data.ipynb saves under Stock/
LOG_DIR    = os.getenv("LOG_DIR",    "logs")
MANIFEST   = pathlib.Path("artifacts/manifest.json")

# Ensure log dir exists
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
@st.cache_resource
def load_manifest():
    """Read artifacts/manifest.json if present; else infer from models/*.pkl."""
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text()).get("tickers", [])
        except Exception:
            pass
    return sorted([pathlib.Path(p).stem for p in glob.glob(f"{MODELS_DIR}/*.pkl")])

@st.cache_resource
def load_model(ticker: str):
    """Load a single SARIMAX model <TICKER>.pkl from models dir."""
    p = pathlib.Path(MODELS_DIR) / f"{ticker}.pkl"
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None

def find_stock_csv(ticker: str):
    """Try to locate the CSV with history to overlay the forecast.
    Searches Stock/Large_Cap, Stock/Mid_Cap, Stock/Small_Cap, then Stock/.
    """
    for grp in ("Large_Cap","Mid_Cap","Small_Cap"):
        p = pathlib.Path(STOCK_DIR) / grp / f"{ticker}.csv"
        if p.exists():
            return p
    p = pathlib.Path(STOCK_DIR) / f"{ticker}.csv"
    return p if p.exists() else None

def log_prediction(ticker, horizon, vals):
    row = {
        "ticker": ticker,
        "horizon": int(horizon),
        "prediction_mean": float(np.mean(vals)) if len(vals)>0 else np.nan,
        "ts": pd.Timestamp.utcnow().isoformat()
    }
    out = pathlib.Path(LOG_DIR)/"predictions.csv"
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)

# ---------- UI ----------
st.set_page_config(page_title="SARIMAX Stock Forecast", layout="wide")
st.title("📈 SARIMAX Stock Forecast (models from `models/`)")

tickers = load_manifest()
if not tickers:
    st.error("No models found in `models/`. Train in stock_mlops.ipynb, then build artifacts/manifest.json.")
    st.stop()

c1,c2 = st.columns([2,1])
with c1:
    ticker = st.selectbox("Ticker", tickers, index=0)
with c2:
    horizon = st.number_input("Forecast horizon (days)", 1, 30, 7, 1)

if st.button("🔮 Forecast"):
    model = load_model(ticker)
    if model is None:
        st.error(f"No model file found for {ticker} in `{MODELS_DIR}`.")
        st.stop()
    try:
        fc = model.forecast(steps=int(horizon))
        vals = fc.values if hasattr(fc, "values") else np.asarray(fc).reshape(-1)
        log_prediction(ticker, horizon, vals)

        stock_csv = find_stock_csv(ticker)
        fig, ax = plt.subplots(figsize=(11,4))

        if stock_csv:
            try:
                hist = pd.read_csv(stock_csv, parse_dates=["Date"], infer_datetime_format=True)
                # normalize columns in case they are lowercase
                if "Date" not in hist.columns:
                    for c in hist.columns:
                        if c.lower() == "date":
                            hist = hist.rename(columns={c:"Date"})
                if "Close" not in hist.columns:
                    for c in hist.columns:
                        if c.lower().replace(" ", "") in ("close","adjclose","adj_close"):
                            hist = hist.rename(columns={c:"Close"})
                hist = hist.sort_values("Date")
                last = hist["Date"].max()
                ax.plot(hist["Date"].tail(120), hist["Close"].tail(120), label="History")
                future_idx = pd.date_range(last, periods=int(horizon)+1, freq="D")[1:]
                ax.plot(future_idx, vals, "--", marker="o", label="Forecast")
                ax.set_xlabel("Date")
            except Exception:
                ax.plot(np.arange(1, len(vals)+1), vals, "--", marker="o", label="Forecast")
                ax.set_xlabel("Step")
        else:
            ax.plot(np.arange(1, len(vals)+1), vals, "--", marker="o", label="Forecast")
            ax.set_xlabel("Step")

        ax.set_title(f"{ticker}: {horizon}-day SARIMAX forecast")
        ax.set_ylabel("Close")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Forecast error: {e}")

st.header("📊 Monitoring")
log_path = pathlib.Path(LOG_DIR)/"predictions.csv"
if log_path.exists():
    logs = pd.read_csv(log_path, parse_dates=["ts"], infer_datetime_format=True)
    st.metric("Total forecasts", len(logs))
    try:
        hourly = logs.set_index("ts").resample("1H").size().rename("req_per_hour").reset_index()
        st.line_chart(hourly.set_index("ts"))
    except Exception:
        pass
    st.dataframe(logs.tail(50))
else:
    st.info("No logs yet. Run a few forecasts.")