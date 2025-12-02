# Optional helper: mirror any flat Stock/<TICKER>.csv into Stock/<CapGroup>/<TICKER>.csv
# and normalize headers to Date/Close (without touching your notebooks).
import os, pandas as pd, pathlib as p

large_caps = ["AAPL","MSFT","AMZN","GOOGL","GOOG","META","NVDA","TSLA","BRK.B","JNJ",
              "JPM","V","MA","PG","XOM","CVX","HD","WMT","KO","PEP","ABBV","AVGO",
              "LLY","UNH","MRK","CSCO","ORCL","NFLX","CRM","COST","DIS","ACN",
              "TXN","INTC","NKE","UPS","IBM","QCOM","DHR","AMGN","GE","CAT","HON",
              "LIN","BLK","ADBE","SPGI","MS","BAC"]
mid_caps = ["ALGN","FICO","WDAY","OKTA","MTCH","ETSY","CHRW","SWKS","DKNG","TDOC","DOCU","Z",
            "PINS","HUBS","DDOG","NET","MDB","ZS","TEAM","LULU","RH","ENPH","FSLR","ROKU",
            "CPRI","BURL","MHK","HAS","ALV","HELE","AAP","CNP","GNRC","RHI","HOLX","BRO",
            "WEX","FFIV","MAN","AER","AOS","CAR","CORT","TPR","MAS","SEE"]
small_caps = ["PLUG","FUBO","EXAS","NTLA","WKHS","CHPT","FSLY","TLRY","NIO","JMIA","COHR",
              "CPIX","BLNK","AEO","FLGT","VRDN","TGNA","KLIC","BLMN","SONO","MGNI","WK",
              "CORT","CALM","TRUP","TNDM","IRBT","BYND","PRPL"]

def normalize_and_copy(ticker, group):
    srcs = [p.Path(f"Stock/{ticker}.csv"),
            p.Path(f"Stock/{group}/{ticker}.csv")]
    dst = p.Path(f"Stock/{group}/{ticker}.csv")
    if dst.exists():
        return
    src = next((s for s in srcs if s.exists()), None)
    if not src:
        print(f"[skip] no CSV for {ticker}")
        return
    try:
        df = pd.read_csv(src)
    except Exception as e:
        print(f"[warn] could not read {src}: {e}")
        return
    # normalize headers: want Date + Close for plotting
    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("date","ds","timestamp","time","time_published"):
            rename[c] = "Date"
        if lc in ("close","adj close","adj_close","adjclose"):
            rename[c] = "Close"
    df = df.rename(columns=rename)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print("→", dst)

for t in large_caps: normalize_and_copy(t, "Large_Cap")
for t in mid_caps:   normalize_and_copy(t, "Mid_Cap")
for t in small_caps: normalize_and_copy(t, "Small_Cap")
print("Done.")