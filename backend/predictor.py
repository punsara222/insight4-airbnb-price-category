"""Loads the trained artifacts, validates user input, and makes predictions.

Flow for one request:
    user JSON -> validate -> fill optional fields with training defaults
    -> preprocessing_pipeline.transform() -> reindex to feature_columns order
    -> final_model.predict()
"""
from pathlib import Path
import json
import logging

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger("insight4")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Fields the user MUST fill in. Everything else is optional and falls back to
# the training median (numbers) or most common value (categories).
# EDIT THIS LIST to match the column names printed by the Colab script.
REQUIRED_FIELDS = [
    "neighbourhood_group_cleansed",   # borough
    "neighbourhood_cleansed",         # neighbourhood
    "room_type",
    "property_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
]

TRUE_WORDS = {"true", "t", "1", "yes", "y"}
FALSE_WORDS = {"false", "f", "0", "no", "n"}


class InputError(Exception):
    """Raised when user input is invalid. Holds a list of readable messages."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(errors))


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model = joblib.load(model_dir / "final_model.pkl")
        self.pre = joblib.load(model_dir / "preprocessing_pipeline.pkl")
        self.feature_columns = list(joblib.load(model_dir / "feature_columns.pkl"))
        self.spec = json.loads((model_dir / "input_spec.json").read_text())
        self.defaults = json.loads((model_dir / "defaults.json").read_text())

        # Optional: neighbourhood -> borough + median coordinates
        lookup_path = model_dir / "neighbourhood_lookup.json"
        self.lookup = json.loads(lookup_path.read_text()) if lookup_path.exists() else {}
        if not self.lookup:
            log.warning("neighbourhood_lookup.json not found - coordinates will use city-wide defaults")

        self.raw_columns = list(self.pre.feature_names_in_)
        self.out_names = list(self.pre.get_feature_names_out())
        self.classes = [str(c) for c in self.model.classes_]

        # Startup checks: fail loudly now rather than give wrong answers later.
        problems = []
        missing_out = [c for c in self.feature_columns if c not in self.out_names]
        if missing_out:
            problems.append(f"feature_columns not produced by the preprocessor: {missing_out}")
        no_spec = [c for c in self.raw_columns if c not in self.spec]
        if no_spec:
            problems.append(f"raw columns with no rule in input_spec.json: {no_spec}")
        bad_required = [c for c in REQUIRED_FIELDS if c not in self.spec]
        if bad_required:
            problems.append(f"REQUIRED_FIELDS not in the model's inputs: {bad_required}")
        if problems:
            raise RuntimeError("Model artifacts do not match:\n- " + "\n- ".join(problems))

        log.info("Model loaded: %s, %d raw inputs, %d model features, classes=%s",
                 type(self.model).__name__, len(self.raw_columns),
                 len(self.feature_columns), self.classes)

    # ---------------------------------------------------------------- validation
    def validate(self, payload: dict):
        """Return (clean_row, fields_filled_with_defaults) or raise InputError."""
        if not isinstance(payload, dict) or not payload:
            raise InputError(["'features' must be a non-empty object of field: value pairs."])

        errors, row, used_defaults = [], {}, []

        unknown = sorted(k for k in payload if k not in self.spec)
        if unknown:
            errors.append(f"Unknown field(s): {', '.join(unknown)}. See GET /schema for valid fields.")

        for col in self.raw_columns:
            rule = self.spec[col]
            val = payload.get(col)

            # Missing value
            if val is None or (isinstance(val, str) and val.strip() == ""):
                if col in REQUIRED_FIELDS:
                    errors.append(f"'{col}' is required.")
                else:
                    row[col] = self.defaults[col]
                    used_defaults.append(col)
                continue

            kind = rule["type"]
            if kind == "numeric":
                if isinstance(val, bool):
                    errors.append(f"'{col}' must be a number, not true/false.")
                    continue
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    errors.append(f"'{col}' must be a number (got {val!r}).")
                    continue
                if not np.isfinite(num):
                    errors.append(f"'{col}' must be a finite number.")
                    continue
                lo, hi = rule["min"], rule["max"]
                if num < lo or num > hi:
                    errors.append(f"'{col}' must be between {lo:g} and {hi:g} (got {num:g}).")
                    continue
                row[col] = num

            elif kind == "boolean":
                if isinstance(val, bool):
                    row[col] = val
                elif str(val).strip().lower() in TRUE_WORDS:
                    row[col] = True
                elif str(val).strip().lower() in FALSE_WORDS:
                    row[col] = False
                else:
                    errors.append(f"'{col}' must be true or false (got {val!r}).")

            else:  # categorical
                text = str(val).strip()
                allowed = rule.get("allowed")
                if allowed is not None and text not in allowed:
                    preview = ", ".join(allowed[:8]) + (" ..." if len(allowed) > 8 else "")
                    errors.append(f"'{col}' has an unknown value {text!r}. Allowed: {preview}")
                    continue
                row[col] = text

        # Cross-field checks using the neighbourhood lookup
        hood = row.get("neighbourhood_cleansed")
        info = self.lookup.get(hood) if hood is not None else None
        if info:
            borough = row.get("neighbourhood_group_cleansed")
            if borough is not None and borough != info["borough"]:
                errors.append(f"Neighbourhood '{hood}' is in {info['borough']}, not {borough}.")
            # If coordinates weren't given, use the neighbourhood's centre instead of a city-wide median
            for coord in ("latitude", "longitude"):
                if coord in used_defaults and coord in row:
                    row[coord] = info[coord]
                    used_defaults.remove(coord)

        if errors:
            raise InputError(errors)
        return row, used_defaults

    # ---------------------------------------------------------------- prediction
    def predict(self, payload: dict) -> dict:
        row, used_defaults = self.validate(payload)

        df = pd.DataFrame([row], columns=self.raw_columns)
        Xt = self.pre.transform(df)
        if hasattr(Xt, "toarray"):  # sparse output from OneHotEncoder
            Xt = Xt.toarray()
        Xt = pd.DataFrame(Xt, columns=self.out_names).reindex(columns=self.feature_columns)

        label = str(self.model.predict(Xt)[0])
        result = {"price_category": label, "used_defaults_for": used_defaults}

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(Xt)[0]
            result["probabilities"] = {c: round(float(p), 4) for c, p in zip(self.classes, probs)}
            result["confidence"] = round(float(max(probs)), 4)
        return result