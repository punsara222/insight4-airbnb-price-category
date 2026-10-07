import json
import requests

BASE_URL = "http://127.0.0.1:8000"
samples = json.load(open("models/sample_requests.json"))

print("=== Known-answer test cases ===")
for i, s in enumerate(samples):
    r = requests.post(f"{BASE_URL}/predict", json={"features": s['features']})
    result = r.json()
    print(f"Test {i}: expected={s['expected']}, got={result}, status={r.status_code}")

print("\n=== Invalid input test cases ===")
invalid_cases = [
    {"features": {}},  # empty
    {"features": {"accommodates": -5}},  # negative value
    {"features": {"room_type": "Castle"}},  # unknown category
]
for i, case in enumerate(invalid_cases):
    r = requests.post(f"{BASE_URL}/predict", json=case)
    print(f"Invalid test {i}: status={r.status_code}, response={r.json()}")