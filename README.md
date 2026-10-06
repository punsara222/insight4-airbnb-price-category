# Insight4 - NYC Airbnb Price Category Predictor

IT3051 Fundamentals of Data Mining - Mini Project 2026 (SLIIT)

Predicts whether a New York City Airbnb listing falls in the **Low, Medium or High** price
category, using an Inside Airbnb NYC listings snapshot. Final model: **Random Forest (tuned)**,
selected on macro F1.

## Team
| Student ID | Contribution |
|---|---|
| IT23642232 | Preprocessing pipeline, Logistic Regression, GitHub repo, FastAPI backend |
| IT23599154 | Decision Tree, city-wide vs borough comparison, presentation |
| IT23580176 | Random Forest (final model), testing |
| IT23728844 | KNN, Streamlit frontend |

## Repository structure
```
backend/     FastAPI service (app.py = endpoints, predictor.py = validation + prediction)
frontend/    Streamlit user interface
models/      final_model.pkl, preprocessing_pipeline.pkl, feature_columns.pkl,
             input_spec.json, defaults.json, sample_requests.json, neighbourhood_lookup.json
tools/       build_neighbourhood_lookup.py
notebooks/   EDA + preprocessing, model development + optimisation
results/     model comparison results and plots
tests/       smoke test for the API
docs/        dataset proposal, technical report, presentation
data/        dataset source + citation (full CSV not committed)
```

## How to run
Requires **Python 3.12** (the pinned numpy / scikit-learn versions have no 3.14 builds).
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# one-time: neighbourhood -> borough + coordinates lookup (needs the training CSV in data/)
python tools/build_neighbourhood_lookup.py data/listings_after_member3.csv

uvicorn backend.app:app --reload        # API on http://127.0.0.1:8000  (docs at /docs)
python tests/smoke_test.py              # in a second terminal: checks the API
streamlit run frontend/app.py           # user interface
```

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Is the API up, which model is loaded |
| GET | `/schema` | Valid fields, allowed values, numeric ranges, required fields |
| POST | `/predict` | Body `{"features": {...}}` returns price category + probabilities |

Invalid input returns HTTP 422 with `{"message": ..., "errors": [...]}`.

## Dataset
Inside Airbnb - New York City listings. http://insideairbnb.com/get-the-data/