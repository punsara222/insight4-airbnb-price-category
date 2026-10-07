import joblib

model = joblib.load("models/final_model.pkl")
print("n_estimators:", model.n_estimators)

if model.n_estimators == 100:
    print("This is the BASELINE model")
elif model.n_estimators == 200:
    print("This is the TUNED model")
else:
    print("Unexpected value — something's off, check manually")