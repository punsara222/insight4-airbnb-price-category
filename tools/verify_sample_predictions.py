import json
import joblib
import pandas as pd

model = joblib.load("models/final_model.pkl")
pre = joblib.load("models/preprocessing_pipeline.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

samples = json.load(open("models/sample_requests.json"))

correct = 0
for i, s in enumerate(samples):
    df = pd.DataFrame([s['features']])
    Xt = pre.transform(df)
    Xt = Xt.toarray() if hasattr(Xt, 'toarray') else Xt
    Xt = pd.DataFrame(Xt, columns=pre.get_feature_names_out()).reindex(columns=feature_columns)
    pred = model.predict(Xt)[0]
    match = pred == s['expected']
    correct += match
    print(f"Sample {i}: expected={s['expected']}, predicted={pred}, match={match}")

print(f"\n{correct}/{len(samples)} samples matched expected output")