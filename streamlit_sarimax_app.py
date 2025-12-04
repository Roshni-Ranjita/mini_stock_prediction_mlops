#!/usr/bin/env python3
import os, glob, json, pathlib, re
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Config ----------
MODELS_DIR = os.getenv("MODELS_DIR", "models")
STOCK_DIR  = os.getenv("STOCK_DIR",  "Stock")
LOG_DIR    = os.getenv("LOG_DIR",    "logs")
MANIFEST   = pathlib.Path("artifacts/manifest.json")
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# allow BRK.B etc., disallow version suffixes like _v3
TICKER_RE = re.compile(r"^[A-Z][A-Z\.]*$")

# ---------- Helpers ----------
@st.cache_resource
def discover_tickers():
    """Prefer manifest; else derive from models/*.pkl, and filter ‘versioned’ names."""
    tickers = []
    if MANIFEST.exists():
        try:
            items = json.loads(MANIFEST.read_text()).get("tickers", [])
            tickers = [t for t in items if TICKER_RE.match(pathlib.Path(t).stem or t)]
        except Exception:
            pass
    if not tickers:  # fallback from models dir
        for p in glob.glob(f"{MODELS_DIR}/*.pkl"):
            name = pathlib.Path(p).stem
            if TICKER_RE.match(name):
                tickers.append(name)
    return sorted(set(tickers))

@st.cache_resource
def load_model(ticker: str):
    p = pathlib.Path(MODELS_DIR) / f"{ticker}.pkl"
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None

def find_stock_csv(ticker: str):
    for grp in ("Large_Cap","Mid_Cap","Small_Cap"):
        p = pathlib.Path(STOCK_DIR) / grp / f"{ticker}.csv"
        if p.exists():
            return p
    # fallback flat layout
    p = pathlib.Path(STOCK_DIR) / f"{ticker}.csv"
    return p if p.exists() else None

def load_history_df(ticker: str) -> pd.DataFrame | None:
    path = find_stock_csv(ticker)
    if not path:
        return None
    df = pd.read_csv(path)
    # normalize headers
    rename = {}
    for c in df.columns:
        lc = c.lower().strip().replace(" ", "")
        if lc in ("date","ds","timestamp","time"):
            rename[c] = "Date"
        if lc in ("close","adjclose","adj_close"):
            rename[c] = "Close"
        if lc == "open":
            rename[c] = "Open"
        if lc == "volume":
            rename[c] = "Volume"
    df = df.rename(columns=rename)
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df

def last_snapshot(df: pd.DataFrame) -> dict:
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row
    close = float(row.get("Close", np.nan))
    openv = float(row.get("Open", np.nan)) if "Open" in df.columns else np.nan
    vol = int(row.get("Volume", np.nan)) if "Volume" in df.columns and not pd.isna(row["Volume"]) else None
    delta = close - float(prev.get("Close", close))
    pct = (delta / float(prev.get("Close", close))) * 100 if prev.get("Close", 0) else np.nan
    return {"date": row["Date"], "open": openv, "close": close, "volume": vol, "delta": delta, "pct": pct}

def log_prediction(ticker, horizon, vals):
    out = pathlib.Path(LOG_DIR)/"predictions.csv"
    pd.DataFrame([{
        "ticker": ticker,
        "horizon": int(horizon),
        "prediction_mean": float(np.mean(vals)) if len(vals)>0 else np.nan,
        "ts": pd.Timestamp.utcnow().isoformat()
    }]).to_csv(out, mode="a", header=not out.exists(), index=False)

# ---------- UI ----------
st.set_page_config(page_title="Smart Stock Forecast", layout="wide")
st.markdown("<h1 style='margin-bottom:0'>🧠 Smart Stock Dashboard</h1><small>Demo — educational only</small>", unsafe_allow_html=True)

tickers = discover_tickers()
if not tickers:
    st.error("No models found. Ensure `models/<TICKER>.pkl` exists. Then run `python build_manifest.py`.")
    st.stop()

# Controls
left, right = st.columns([2, 1])
with left:
    ticker = st.selectbox("Select Ticker", tickers, index=0)
with right:
    horizon = st.slider("Forecast Horizon (days)", 1, 30, 7, 1)

# KPI cards (last day snapshot)
hist = load_history_df(ticker)
if hist is not None and len(hist) >= 1:
    snap = last_snapshot(hist)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"${snap['close']:.2f}", f"{snap['delta']:+.2f}")
    c2.metric("Last Open",  f"${snap['open']:.2f}" if not np.isnan(snap['open']) else "—")
    c3.metric("Change %",   f"{snap['pct']:+.2f}%" if not np.isnan(snap['pct']) else "—")
    c4.metric("Volume",     f"{snap['volume']:,}" if snap['volume'] is not None else "—")
else:
    st.info("No historical CSV located under Stock/; forecast still works.")

# Tabs
tab1, tab2 = st.tabs(["🔮 Forecast", "📊 Monitoring"])

with tab1:
    colA, colB = st.columns([1,1])
    go = colA.button("Run Forecast", type="primary", use_container_width=True)
    if go:
        model = load_model(ticker)
        if model is None:
            st.error(f"Model file not found for {ticker} in `{MODELS_DIR}`.")
        else:
            try:
                fc = model.forecast(steps=int(horizon))
                vals = fc.values if hasattr(fc, "values") else np.asarray(fc).reshape(-1)
                log_prediction(ticker, horizon, vals)

                # simple demo signal
                last_close = hist["Close"].iloc[-1] if hist is not None else np.nan
                mean_fc = float(np.mean(vals)) if len(vals) else np.nan
                signal = "Bullish 📈" if (not np.isnan(last_close) and mean_fc > last_close) else "Bearish 📉"
                colB.success(f"Signal (demo): {signal}")

                # chart: last 180d + MA20/MA50 + forecast
                fig, ax = plt.subplots(figsize=(11,4))
                if hist is not None and len(hist) > 0:
                    tail = hist.tail(180).copy()
                    tail["MA20"] = tail["Close"].rolling(20).mean()
                    tail["MA50"] = tail["Close"].rolling(50).mean()
                    ax.plot(tail["Date"], tail["Close"], label="Close", alpha=0.9)
                    if tail["MA20"].notna().any(): ax.plot(tail["Date"], tail["MA20"], label="MA20", alpha=0.8)
                    if tail["MA50"].notna().any(): ax.plot(tail["Date"], tail["MA50"], label="MA50", alpha=0.8)
                    last = tail["Date"].max()
                    future_idx = pd.date_range(last, periods=int(horizon)+1, freq="D")[1:]
                    ax.plot(future_idx, vals, "--", marker="o", label="Forecast")
                    ax.set_xlabel("Date")
                else:
                    ax.plot(np.arange(1, len(vals)+1), vals, "--", marker="o", label="Forecast")
                    ax.set_xlabel("Step")
                ax.set_title(f"{ticker}: {horizon}-day Forecast")
                ax.set_ylabel("Close")
                ax.legend()
                st.pyplot(fig)

                st.caption("Signals are illustrative only — not financial advice.")
            except Exception as e:
                st.error(f"Forecast error: {e}")

with tab2:
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