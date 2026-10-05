"""Quick end-to-end check. Start the backend first, then run:
    python tests/smoke_test.py
"""
import json
from pathlib import Path

import requests

API = "http://127.0.0.1:8000"
samples = json.loads((Path(__file__).resolve().parent.parent / "models" / "sample_requests.json").read_text())

print("Health:", requests.get(f"{API}/health").json())

# 1. Real rows: the API must give the same answer as Colab did
passed = 0
for i, s in enumerate(samples, 1):
    r = requests.post(f"{API}/predict", json={"features": s["features"]})
    got = r.json().get("price_category")
    ok = got == s["expected"]
    passed += ok
    print(f"Sample {i}: expected={s['expected']:<8} got={got:<8} {'PASS' if ok else 'FAIL'}")
print(f"{passed}/{len(samples)} match Colab\n")

# 2. Bad inputs: each should return 422 with a clear message
bad_cases = {
    "empty features": {"features": {}},
    "missing body key": {"wrong": {}},
    "text in number field": {"features": {**samples[0]["features"], "accommodates": "four"}},
    "unknown category": {"features": {**samples[0]["features"], "room_type": "Spaceship"}},
    "unknown field": {"features": {**samples[0]["features"], "colour": "blue"}},
}
for name, body in bad_cases.items():
    r = requests.post(f"{API}/predict", json=body)
    print(f"{name:22s} -> {r.status_code} {r.json()}")
