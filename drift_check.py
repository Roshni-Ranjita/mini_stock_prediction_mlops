#!/usr/bin/env python3
# Batch drift checker with MLflow logging (robust PSI + safe conversions)
import os, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance, levene
import mlflow
import joblib

MODELS_DIR = os.getenv("MODELS_DIR", "models")
STOCK_DIR  = os.getenv("STOCK_DIR",  "Stock")
MLRUNS_URI = os.getenv("MLFLOW_URI", "file:./mlruns")
BASELINE_START = os.getenv("BASELINE_START", "2020-01-01")
BASELINE_END   = os.getenv("BASELINE_END",   "2022-12-31")
RECENT_DAYS    = int(os.getenv("RECENT_DAYS", "30"))

def psi(expected, actual, bins=10):
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

def load_hist_csv(stock_dir, ticker):
    candidates = [
        Path(stock_dir)/"Large_Cap"/f"{ticker}.csv",
        Path(stock_dir)/"Mid_Cap"/f"{ticker}.csv",
        Path(stock_dir)/"Small_Cap"/f"{ticker}.csv",
        Path(stock_dir)/f"{ticker}.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            rename = {}
            for c in df.columns:
                lc = str(c).lower().strip().replace(" ", "")
                if lc in ("date","ds","timestamp","time"):
                    rename[c] = "Date"
                if lc in ("close","adjclose","adj_close"):
                    rename[c] = "Close"
            df = df.rename(columns=rename)
            if "Date" in df.columns and "Close" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                return df.sort_values("Date").dropna(subset=["Date","Close"])
    return None

def compute_drift_for_ticker(ticker):
    out = {"ticker": ticker, "status": "ok"}
    df = load_hist_csv(STOCK_DIR, ticker)
    if df is None or len(df) < 250:
        out["status"] = "no_data"
        return out

    df["return_1d"] = df["Close"].pct_change()
    base = df[(df["Date"]>=BASELINE_START) & (df["Date"]<=BASELINE_END)].copy()
    recent = df.tail(RECENT_DAYS).copy()
    if len(base) < 100 or len(recent) < 5:
        out["status"] = "insufficient_window"
        return out

    b = base["return_1d"].dropna(); r = recent["return_1d"].dropna()
    if len(b)>20 and len(r)>5:
        out["psi_return_1d"] = psi(b, r)
        out["ks_p_return_1d"] = float(ks_2samp(b, r).pvalue)
        out["emd_return_1d"] = float(wasserstein_distance(b, r))
        out["vol_ratio_return_1d"] = float(r.std(ddof=1) / max(b.std(ddof=1), 1e-9))

    model_path = Path(MODELS_DIR)/f"{ticker}.pkl"
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            y = _to_arr(recent["Close"].astype(float))
            try:
                predobj = model.get_prediction(start=len(df)-RECENT_DAYS, end=len(df)-1)
                preds = _to_arr(predobj.predicted_mean)
            except Exception:
                fc = model.forecast(steps=len(recent))
                preds = _to_arr(fc)
            m = min(len(y), len(preds))
            if m >= 3:
                y = y[:m]; preds = preds[:m]
                resid = y - preds
                out["rmse_recent"] = float(np.sqrt(np.mean(resid**2)))

                b_close = _to_arr(base["Close"].tail(m))
                try:
                    b_predobj = model.get_prediction(start=len(base)-m, end=len(base)-1)
                    b_preds   = _to_arr(b_predobj.predicted_mean)
                except Exception:
                    b_preds   = b_close
                b_resid = b_close - b_preds[:m]
                out["rmse_baseline"] = float(np.sqrt(np.mean(b_resid**2)))
                out["rmse_ratio"] = float(out["rmse_recent"] / max(out["rmse_baseline"], 1e-9))
                out["resid_levene_p"] = float(levene(resid, b_resid, center='median').pvalue)
        except Exception as e:
            out["model_error"] = str(e)[:200]

    flags = []
    if out.get("psi_return_1d", 0) > 0.25: flags.append("PSI>0.25")
    if out.get("ks_p_return_1d", 1.0) < 0.01: flags.append("KS p<0.01")
    vr = out.get("vol_ratio_return_1d", 1.0)
    if vr > 1.5 or vr < 0.67: flags.append("Vol drift")
    if out.get("rmse_ratio", 1.0) > 1.5: flags.append("Error↑")
    if out.get("resid_levene_p", 1.0) < 0.01: flags.append("Residual var drift")
    out["alerts"] = "; ".join(flags)
    return out

def main():
    mlflow.set_tracking_uri(MLRUNS_URI)
    mlflow.set_experiment("drift_monitoring")
    # exclude obvious versioned model names
    tickers = sorted([Path(p).stem for p in glob.glob(f"{MODELS_DIR}/*.pkl")
                      if not Path(p).stem.endswith(tuple([f"_v{i}" for i in range(1,50)]))])
    rows = []
    with mlflow.start_run(run_name="daily_drift_check"):
        for t in tickers:
            r = compute_drift_for_ticker(t)
            rows.append(r)
            for k in ["psi_return_1d","ks_p_return_1d","vol_ratio_return_1d","rmse_ratio","resid_levene_p"]:
                if k in r and pd.notna(r[k]):
                    mlflow.log_metric(f"{t}_{k}", float(r[k]))
            if r.get("alerts"):
                mlflow.set_tag(f"{t}_alerts", r["alerts"])
    os.makedirs("logs", exist_ok=True)
    pd.DataFrame(rows).to_csv("logs/drift.csv", index=False)
    print("Wrote logs/drift.csv")

if __name__ == "__main__":
    main()