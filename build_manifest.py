import json, glob, os
os.makedirs("artifacts", exist_ok=True)
tickers = sorted([os.path.basename(p).replace(".pkl","") for p in glob.glob("models/*.pkl")])
with open("artifacts/manifest.json","w") as f:
    json.dump({"tickers": tickers}, f, indent=2)
print("Wrote artifacts/manifest.json with", len(tickers), "tickers")