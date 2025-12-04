#!/usr/bin/env python3
import json, glob, os, re, pathlib
TICKER_RE = re.compile(r"^[A-Z][A-Z\.]*$")
os.makedirs("artifacts", exist_ok=True)
names = []
for p in glob.glob("models/*.pkl"):
    name = pathlib.Path(p).stem
    if TICKER_RE.match(name) and not re.search(r"_v\d+$", name):
        names.append(name)
names = sorted(set(names))
with open("artifacts/manifest.json","w") as f:
    json.dump({"tickers": names}, f, indent=2)
print("Wrote artifacts/manifest.json with", len(names), "tickers")