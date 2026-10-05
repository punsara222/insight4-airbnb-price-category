# ============================================================
# RUN THIS IN COLAB (paste into ONE cell) - prepares everything the backend needs.
# Everything to download ends up in: /content/drive/MyDrive/Insight4_Airbnb/repo_models/
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os, json, joblib, shutil
import numpy as np, pandas as pd, sklearn

BASE = '/content/drive/MyDrive/Insight4_Airbnb'
ART  = BASE + '/backend_artifacts'              # where Step 9 saved the 3 .pkl files
RAW  = BASE + '/listings_after_member3.csv'     # data the preprocessor was fitted on (change path if different)
OUT  = BASE + '/repo_models'                    # download this whole folder into the repo's models/ folder
os.makedirs(OUT, exist_ok=True)

# ---- 0. Copy the 3 files from Colab's temporary disk to Drive (safe storage)
os.makedirs(ART, exist_ok=True)
for f in ['final_model.pkl', 'feature_columns.pkl', 'preprocessing_pipeline.pkl']:
    if os.path.exists('/content/backend/' + f):
        shutil.copy('/content/backend/' + f, ART + '/' + f)
print('backend_artifacts in Drive:', os.listdir(ART))

# ---- 1. Load the three artifacts --------------------------------------
pre             = joblib.load(ART + '/preprocessing_pipeline.pkl')
feature_columns = list(joblib.load(ART + '/feature_columns.pkl'))
model           = joblib.load(ART + '/final_model.pkl')
# If the winner was saved as a GridSearchCV object, keep only the trained model inside it
model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
print('Model type :', type(model).__name__)
print('Classes    :', list(model.classes_))

raw_cols  = list(pre.feature_names_in_)        # columns the USER INPUT must provide
out_names = list(pre.get_feature_names_out())  # columns the preprocessor produces
print(f'\nPreprocessor expects {len(raw_cols)} raw columns:\n', raw_cols)

# ---- 2. Check the reindex step will work ------------------------------
missing_out = [c for c in feature_columns if c not in out_names]
print('\nfeature_columns missing from preprocessor output:', missing_out or 'none (good)')
dropped = [c for c in out_names if c not in feature_columns]
print('Preprocessor outputs removed by reindex (e.g. leakage drops):', dropped or 'none')
assert not missing_out, 'STOP: column names do not match - send this output to Claude.'

# ---- 3. Check the raw data has every column ---------------------------
raw = pd.read_csv(RAW)
missing_raw = [c for c in raw_cols if c not in raw.columns]
print('\nRaw columns not in the CSV (created later?):', missing_raw or 'none (good)')
assert not missing_raw, 'STOP: these columns were engineered after member 3 - send this output to Claude.'
X = raw[raw_cols]

# ---- 4. Build input_spec.json (validation rules) and defaults.json ----
spec, defaults = {}, {}
for c in raw_cols:
    s = X[c]
    nonnull = s.dropna()
    if pd.api.types.is_bool_dtype(s):
        spec[c] = {'type': 'boolean'}
        defaults[c] = bool(nonnull.mode().iloc[0]) if len(nonnull) else False
    elif pd.api.types.is_numeric_dtype(s):
        spec[c] = {'type': 'numeric', 'min': float(nonnull.min()), 'max': float(nonnull.max())}
        defaults[c] = float(nonnull.median())
    else:
        vals = sorted(nonnull.astype(str).unique().tolist())
        spec[c] = {'type': 'categorical', 'allowed': vals if len(vals) <= 500 else None}
        defaults[c] = str(nonnull.astype(str).mode().iloc[0]) if len(nonnull) else ''
json.dump(spec, open(OUT + '/input_spec.json', 'w'), indent=2)
json.dump(defaults, open(OUT + '/defaults.json', 'w'), indent=2)

# ---- 5. Copy artifacts; save the model compressed (same model, smaller file)
shutil.copy(ART + '/preprocessing_pipeline.pkl', OUT + '/preprocessing_pipeline.pkl')
joblib.dump(feature_columns, OUT + '/feature_columns.pkl')
joblib.dump(model, OUT + '/final_model.pkl', compress=3)
model_small = joblib.load(OUT + '/final_model.pkl')

# ---- 6. Sample requests with the EXPECTED answer (used by the smoke test)
def run_pipeline(df, m):
    Xt = pre.transform(df)
    Xt = Xt.toarray() if hasattr(Xt, 'toarray') else Xt
    Xt = pd.DataFrame(Xt, columns=out_names).reindex(columns=feature_columns)
    return m.predict(Xt)

sample = X.dropna().sample(5, random_state=42)
expected = run_pipeline(sample, model)
assert (expected == run_pipeline(sample, model_small)).all(), 'Compressed model gives different answers!'
samples = [{'features': json.loads(r.to_json()), 'expected': str(e)}
           for (_, r), e in zip(sample.iterrows(), expected)]
json.dump(samples, open(OUT + '/sample_requests.json', 'w'), indent=2)

# ---- 7. Versions for requirements.txt + file sizes --------------------
print('\n--- Put these EXACT versions in requirements.txt ---')
print(f'scikit-learn=={sklearn.__version__}')
print(f'numpy=={np.__version__}')
print(f'pandas=={pd.__version__}')
print(f'joblib=={joblib.__version__}')
print('\n--- Files ready in', OUT, '---')
for f in sorted(os.listdir(OUT)):
    print(f'{f:28s} {os.path.getsize(os.path.join(OUT, f)) / 1e6:8.2f} MB')