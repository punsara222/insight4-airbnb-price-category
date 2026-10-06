"""FastAPI backend for the Insight4 Airbnb price-category predictor.

Run from the repo root:
    uvicorn backend.app:app --reload
Then open http://127.0.0.1:8000/docs to try it.
"""
import logging
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .predictor import REQUIRED_FIELDS, InputError, Predictor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("insight4")

app = FastAPI(
    title="Insight4 - NYC Airbnb Price Category API",
    description="Predicts whether a NYC Airbnb listing is Low, Medium or High priced.",
    version="1.0.0",
)

# Allow the Streamlit / web frontend to call this API.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = Predictor()  # loads the model once, at startup


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Listing details as field: value pairs. See GET /schema for valid fields.",
        examples=[{"neighbourhood_group_cleansed": "Brooklyn",
                   "neighbourhood_cleansed": "Williamsburg",
                   "room_type": "Entire home/apt", "property_type": "Entire rental unit",
                   "accommodates": 4, "bathrooms": 1, "bedrooms": 2, "beds": 2}],
    )


@app.exception_handler(RequestValidationError)
async def bad_body_handler(request: Request, exc: RequestValidationError):
    """Clear message when the request body itself is malformed."""
    details = [f"{'/'.join(str(p) for p in e.get('loc', []))}: {e.get('msg')}" for e in exc.errors()]
    return JSONResponse(
        status_code=422,
        content={"message": 'Request body must be JSON like {"features": {"room_type": "...", ...}}',
                 "errors": details},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": type(predictor.model).__name__, "classes": predictor.classes}


@app.get("/schema")
def schema():
    """Everything the frontend needs to build its form."""
    return {
        "required": REQUIRED_FIELDS,
        "fields": predictor.spec,
        "defaults": predictor.defaults,
        "classes": predictor.classes,
        "neighbourhoods_by_borough": _hoods_by_borough(),
    }


def _hoods_by_borough():
    out = {}
    for hood, info in predictor.lookup.items():
        out.setdefault(info["borough"], []).append(hood)
    return {b: sorted(h) for b, h in sorted(out.items())}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return predictor.predict(req.features)
    except InputError as e:
        return JSONResponse(status_code=422, content={"message": "Invalid input", "errors": e.errors})
    except Exception:
        log.exception("Prediction failed")
        return JSONResponse(status_code=500,
                            content={"message": "Prediction failed due to a server error.", "errors": []})