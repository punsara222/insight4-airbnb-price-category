"""Builds models/neighbourhood_lookup.json from the training data.

For every neighbourhood it stores:
  - the borough it belongs to       (used to reject e.g. Brooklyn + Harlem)
  - its median latitude / longitude (used when the user doesn't give coordinates)

Run once from the repo root:
    python tools/build_neighbourhood_lookup.py data/listings_after_member3.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/listings_after_member3.csv"
df = pd.read_csv(csv_path, usecols=["neighbourhood_cleansed", "neighbourhood_group_cleansed",
                                    "latitude", "longitude"]).dropna()

lookup = {}
for hood, g in df.groupby("neighbourhood_cleansed"):
    lookup[str(hood)] = {
        "borough": str(g["neighbourhood_group_cleansed"].mode().iloc[0]),
        "latitude": round(float(g["latitude"].median()), 6),
        "longitude": round(float(g["longitude"].median()), 6),
    }

out = Path(__file__).resolve().parent.parent / "models" / "neighbourhood_lookup.json"
out.write_text(json.dumps(lookup, indent=2))
print(f"Saved {len(lookup)} neighbourhoods to {out}")