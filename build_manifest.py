import json, glob, os, re, pathlib

# Allow tickers like AAPL, MSFT, BRK.B — but exclude versioned names (e.g., AAPL_v3)
TICKER_RE = re.compile(r"^[A-Z][A-Z\.]*$")

os.makedirs("artifacts", exist_ok=True)
names = []
for p in glob.glob("models/*.pkl"):
    name = pathlib.Path(p).stem
    if TICKER_RE.match(name):
        names.append(name)

names = sorted(set(names))
with open("artifacts/manifest.json","w") as f:
    json.dump({"tickers": names}, f, indent=2)
print("Wrote artifacts/manifest.json with", len(names), "tickers")