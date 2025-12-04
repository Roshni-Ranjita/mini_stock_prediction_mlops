#!/usr/bin/env python3
# Smart Stock Dashboard (Streamlit) — WITH Overview tab and unique keys per tab
import os, glob, json, pathlib, re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance, levene

# ---------------- Config ----------------
MODELS_DIR = os.getenv("MODELS_DIR", "models")
STOCK_DIR  = os.getenv("STOCK_DIR",  "Stock")
LOG_DIR    = os.getenv("LOG_DIR",    "logs")
MANIFEST   = pathlib.Path("artifacts/manifest.json")
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

TICKER_RE = re.compile(r"^[A-Z][A-Z\.]*$")   # allow BRK.B

# ---------------- Helpers ----------------
@st.cache_resource
def discover_tickers():
    """Prefer artifacts/manifest.json; fall back to models/*.pkl; drop versioned names like _v1."""
    tickers = []
    if MANIFEST.exists():
        try:
            items = json.loads(MANIFEST.read_text()).get("tickers", [])
            tickers = [t for t in items if TICKER_RE.match(pathlib.Path(t).stem or t)]
        except Exception:
            pass
    if not tickers:
        for p in glob.glob(f"{MODELS_DIR}/*.pkl"):
            name = pathlib.Path(p).stem
            if TICKER_RE.match(name):
                tickers.append(name)
    tickers = [t for t in tickers if not re.search(r"_v\d+$", t)]
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
    p = pathlib.Path(STOCK_DIR) / f"{ticker}.csv"
    return p if p.exists() else None

def load_history_df(ticker: str):
    path = find_stock_csv(ticker)
    if not path:
        return None
    df = pd.read_csv(path)
    # normalize column names
    rename = {}
    for c in df.columns:
        lc = str(c).lower().strip().replace(" ", "")
        if lc in ("date","ds","timestamp","time"):
            rename[c] = "Date"
        if lc in ("close","adjclose","adj_close","closingprice"):
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

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def last_snapshot(df: pd.DataFrame):
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row
    close = _safe_float(row.get("Close", np.nan))
    openv = _safe_float(row.get("Open", np.nan)) if "Open" in df.columns else float("nan")
    vol = row.get("Volume", None)
    try:
        vol = int(vol) if vol is not None and not pd.isna(vol) else None
    except Exception:
        vol = None
    delta = close - _safe_float(prev.get("Close", close))
    denom = _safe_float(prev.get("Close", 0))
    pct = (delta / denom) * 100 if denom else float("nan")
    return {"date": row["Date"], "open": openv, "close": close, "volume": vol, "delta": delta, "pct": pct}

def log_prediction(ticker, horizon, vals):
    out = pathlib.Path(LOG_DIR)/"predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "ticker": ticker,
        "horizon": int(horizon),
        "prediction_mean": float(np.mean(vals)) if len(vals)>0 else float("nan"),
        "ts": pd.Timestamp.utcnow().isoformat()
    }]).to_csv(out, mode="a", header=not out.exists(), index=False)

def psi(expected, actual, bins=10):
    """Robust PSI on 1D arrays/Series; clips zero-prob bins to avoid inf."""
    e = pd.Series(expected).dropna().to_numpy()
    a = pd.Series(actual).dropna().to_numpy()
    if e.size == 0 or a.size == 0:
        return np.nan
    edges = np.unique(np.percentile(e, np.linspace(0, 100, bins+1)))
    if edges.size < 3:
        return np.nan
    e_hist, _ = np.histogram(e, bins=edges)
    a_hist, _ = np.histogram(a, bins=edges)
    e_pct = np.clip(e_hist / max(e_hist.sum(), 1), 1e-6, 1.0)
    a_pct = np.clip(a_hist / max(a_hist.sum(), 1), 1e-6, 1.0)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

def _to_arr(x):
    return x.values if hasattr(x, "values") else np.asarray(x, dtype="float64").reshape(-1)

# ---------------- UI ----------------
st.set_page_config(page_title="Smart Stock Dashboard", layout="wide")
st.markdown("<h1 style='margin-bottom:0'>🧠 Smart Stock Dashboard</h1>", unsafe_allow_html=True)

tickers = discover_tickers()
if not tickers:
    st.error("No models found. Ensure `models/<TICKER>.pkl` exists. Then run `python build_manifest.py`.")
    st.stop()

# ---- shared selection state ----
if "ticker" not in st.session_state:
    st.session_state.ticker = tickers[0]
if "horizon" not in st.session_state:
    st.session_state.horizon = 7

def render_controls(container, show_horizon=True, key_prefix="fc"):
    """Render ticker (and optional horizon) inside a tab; store in session_state.
       key_prefix ensures unique Streamlit widget keys per tab."""
    with container:
        if show_horizon:
            c1, c2 = st.columns([2, 1])
        else:
            c1 = st.container(); c2 = None

        opts = discover_tickers()
        default_idx = opts.index(st.session_state.ticker) if st.session_state.ticker in opts else 0

        sel = c1.selectbox(
            "Select Ticker", opts, index=default_idx,
            key=f"{key_prefix}_ticker_select"
        )
        st.session_state.ticker = sel

        if show_horizon:
            st.session_state.horizon = c2.slider(
                "Forecast Horizon (days)", 1, 30, int(st.session_state.horizon), 1,
                key=f"{key_prefix}_horizon_slider"
            )

    return st.session_state.ticker, st.session_state.horizon

# Tabs: Overview first; controls live only inside Forecast/Drift
tab0, tab1, tab2, tab3 = st.tabs(["🧭 Overview", "🔮 Forecast", "📊 Monitoring", "🧪 Drift"])

# ---------- Overview tab (no controls here) ----------
with tab0:
    st.markdown("""
### What is this dashboard?
This is an **educational stock forecasting & MLOps demo**.  
It loads **historical prices** from your `Stock/` folder and **trained SARIMAX models** from `models/`, then:
- forecasts the next *N* days,
- logs forecasts for **monitoring**, and
- checks **data/model drift**.

**Not financial advice.** Values/alerts are for demonstration.

---

### What can I do here?
1. **Pick a ticker** and a **horizon** in the **Forecast** tab.  
2. **Run Forecast** to see trend (MA20/MA50) + model forecast.  
3. Switch to **Monitoring** to see usage logs.  
4. Switch to **Drift** to compare recent vs baseline.

---

### Sections at a glance
- **🔮 Forecast** – Last Close/Open/Change/Volume + chart with MA20/MA50 + forecast.
- **📊 Monitoring** – Hourly forecast count + latest runs table.
- **🧪 Drift** – Compare **recent 30d** vs **baseline (≤ 2022-12-31)** using:
  - PSI (> **0.25** ⇒ drift),
  - KS p (< **0.01** ⇒ different),
  - Vol ratio (< **0.67** or > **1.5** ⇒ regime change),
  - RMSE ratio (> **1.5** ⇒ model error worsening),
  - Levene p (< **0.01** ⇒ residual variance changed).

---

### Where data comes from
- **Prices**: `Stock/Large_Cap|Mid_Cap|Small_Cap/<TICKER>.csv` or `Stock/<TICKER>.csv` with `Date`, `Close` (optional `Open`, `Volume`).
- **Models**: `models/<TICKER>.pkl`. Dropdown is generated from `artifacts/manifest.json` (`python build_manifest.py`).
""")

# ---------- Forecast tab ----------
with tab1:
    ticker, horizon = render_controls(st.container(), show_horizon=True, key_prefix="fc")

    # KPIs
    hist = load_history_df(ticker)
    if hist is not None and len(hist) >= 1:
        snap = last_snapshot(hist)
        c1, c2, c3, c4 = st.columns(4)
        if not np.isnan(snap['close']):
            c1.metric("Last Close", f"${snap['close']:.2f}", f"{snap['delta']:+.2f}")
        else:
            c1.metric("Last Close", "—")
        c2.metric("Last Open",  f"${snap['open']:.2f}" if not np.isnan(snap['open']) else "—")
        c3.metric("Change %",   f"{snap['pct']:+.2f}%" if not np.isnan(snap['pct']) else "—")
        c4.metric("Volume",     f"{snap['volume']:,}" if snap['volume'] is not None else "—")

    go_col, msg_col = st.columns([1,3])
    go = go_col.button("Run Forecast", type="primary", use_container_width=True)

    if go:
        model = load_model(ticker)
        if model is None:
            st.error(f"Model file not found for {ticker} in `{MODELS_DIR}`.")
        else:
            try:
                fc = model.forecast(steps=int(horizon))
                vals = _to_arr(fc)
                log_prediction(ticker, horizon, vals)

                last_close = hist["Close"].iloc[-1] if hist is not None else float("nan")
                mean_fc = float(np.mean(vals)) if len(vals) else float("nan")
                signal = "Bullish 📈" if (not np.isnan(last_close) and mean_fc > last_close) else "Bearish 📉"
                msg_col.success(f"Signal (demo): {signal}")

                fig, ax = plt.subplots(figsize=(11,4))
                if hist is not None and len(hist) > 0:
                    tail = hist.tail(180).copy()
                    tail["MA20"] = tail["Close"].rolling(20).mean()
                    tail["MA50"] = tail["Close"].rolling(50).mean()
                    ax.plot(tail["Date"], tail["Close"], label="Close")
                    if tail["MA20"].notna().any(): ax.plot(tail["Date"], tail["MA20"], label="MA20")
                    if tail["MA50"].notna().any(): ax.plot(tail["Date"], tail["MA50"], label="MA50")
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

# ---------- Monitoring tab ----------
with tab2:
    log_path = pathlib.Path(LOG_DIR)/"predictions.csv"
    if log_path.exists():
        logs = pd.read_csv(log_path, parse_dates=["ts"], infer_datetime_format=True)
        if "ticker" in logs.columns:
            logs = logs[~logs["ticker"].astype(str).str.contains(r"_v\d+$", regex=True)]
        st.metric("Total forecasts", len(logs))
        try:
            hourly = logs.set_index("ts").resample("1H").size().rename("req_per_hour").reset_index()
            st.line_chart(hourly.set_index("ts"))
        except Exception:
            pass
        st.dataframe(logs.tail(50))
    else:
        st.info("No logs yet. Run a few forecasts.")

# ---------- Drift tab ----------
with tab3:
    ticker, _ = render_controls(st.container(), show_horizon=False, key_prefix="dr")
    st.subheader("Drift checks (last N days vs baseline)")
    cold1, cold2 = st.columns(2)
    recent_days = cold1.slider("Recent window (days)", 7, 90, 30, 1)
    base_end = cold2.text_input("Baseline end (YYYY-MM-DD)", value="2022-12-31")

    try:
        df_hist = load_history_df(ticker)
        if df_hist is None or len(df_hist) < 200:
            st.info("Not enough history to compute drift.")
        else:
            df_hist = df_hist.copy()
            df_hist["return_1d"] = df_hist["Close"].pct_change()

            baseline = df_hist[df_hist["Date"] <= pd.to_datetime(base_end)].copy()
            recent   = df_hist.tail(int(recent_days)).copy()

            if len(baseline) < 100 or len(recent) < 5:
                st.info("Insufficient baseline or recent window.")
            else:
                b = baseline["return_1d"].dropna()
                r = recent["return_1d"].dropna()

                psi_v     = psi(b, r)
                ks_p      = float(ks_2samp(b, r).pvalue) if len(b) and len(r) else np.nan
                _emd      = float(wasserstein_distance(b, r)) if len(b) and len(r) else np.nan
                vol_ratio = float(r.std(ddof=1) / max(b.std(ddof=1), 1e-9)) if len(b) and len(r) else np.nan

                # Model-based metrics
                rmse_ratio = np.nan
                resid_p    = np.nan
                model = load_model(ticker)
                if model is not None and len(recent) >= 3:
                    y = _to_arr(recent["Close"].astype(float))
                    try:
                        predobj = model.get_prediction(start=len(df_hist)-len(recent), end=len(df_hist)-1)
                        preds = _to_arr(predobj.predicted_mean)
                    except Exception:
                        preds = _to_arr(model.forecast(steps=len(recent)))

                    m = min(len(y), len(preds))
                    if m >= 3:
                        y     = y[:m]
                        preds = preds[:m]
                        resid = y - preds
                        rmse_recent = float(np.sqrt(np.mean(resid**2)))

                        b_close = _to_arr(baseline["Close"].tail(m))
                        try:
                            b_predobj = model.get_prediction(start=len(baseline)-m, end=len(baseline)-1)
                            b_preds   = _to_arr(b_predobj.predicted_mean)
                        except Exception:
                            b_preds   = b_close
                        b_resid   = b_close - b_preds[:m]
                        rmse_base = float(np.sqrt(np.mean(b_resid**2)))
                        rmse_ratio = float(rmse_recent / max(rmse_base, 1e-9))
                        resid_p    = float(levene(resid, b_resid, center='median').pvalue)

                # KPIs
                k1,k2,k3,k4,k5 = st.columns(5)
                k1.metric("PSI (returns)", f"{psi_v:.3f}" if pd.notna(psi_v) else "—")
                k2.metric("KS p-value",    f"{ks_p:.3g}"  if pd.notna(ks_p) else "—")
                k3.metric("Vol ratio",     f"{vol_ratio:.2f}" if pd.notna(vol_ratio) else "—")
                k4.metric("RMSE ratio",    f"{rmse_ratio:.2f}" if pd.notna(rmse_ratio) else "—")
                k5.metric("Levene p",      f"{resid_p:.3g}" if pd.notna(resid_p) else "—")

                alerts = []
                if pd.notna(psi_v) and psi_v > 0.25: alerts.append("PSI>0.25")
                if pd.notna(ks_p)  and ks_p  < 0.01: alerts.append("KS p<0.01")
                if pd.notna(vol_ratio) and (vol_ratio > 1.5 or vol_ratio < 0.67): alerts.append("Vol drift")
                if pd.notna(rmse_ratio) and rmse_ratio > 1.5: alerts.append("Error↑")
                if pd.notna(resid_p)    and resid_p < 0.01: alerts.append("Residual var drift")

                (st.error if alerts else st.success)(" | ".join(alerts) if alerts else "No drift flags triggered.")

                # Plots
                fig = plt.figure(figsize=(10,3))
                plt.hist(b, bins=40, alpha=0.6, label="Baseline", density=True)
                plt.hist(r, bins=20, alpha=0.6, label="Recent", density=True)
                plt.title("Return distribution: baseline vs recent")
                plt.xlabel("Daily return"); plt.ylabel("Density"); plt.legend()
                st.pyplot(fig)

                fig2 = plt.figure(figsize=(10,3))
                plt.plot(recent["Date"], recent["return_1d"])
                plt.title("Recent daily returns"); plt.xlabel("Date"); plt.ylabel("Return")
                st.pyplot(fig2)
    except Exception as ex:
        st.error("Drift computation failed.")
        st.exception(ex)