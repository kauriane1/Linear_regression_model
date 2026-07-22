"""
Crop Yield Prediction API
Predicts crop yield (hg/ha) from country, crop, year, rainfall, pesticides, and temperature.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import io

# ---------------------------------------------------------------------------
# Load the trained bundle (model + scaler + encoders) saved from the notebook
# ---------------------------------------------------------------------------
bundle = joblib.load("best_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
area_encoder = bundle["area_encoder"]
item_encoder = bundle["item_encoder"]
feature_order = bundle["feature_order"]

# The valid country and crop names the encoders know about
VALID_AREAS = list(area_encoder.classes_)
VALID_ITEMS = list(item_encoder.classes_)

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Crop Yield Prediction API",
    description="Predicts crop yield (hg/ha) to support food security planning.",
    version="1.0.0",
)

# Explicit, non-wildcard CORS. We allow the origins that actually need access:
# local development and the deployed frontend. We restrict methods to the ones
# the API uses, and allow standard headers. Credentials are disabled because
# the API uses no cookies or auth sessions.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Input schema with types AND realistic range constraints (from the data)
# ---------------------------------------------------------------------------
class YieldInput(BaseModel):
    Area: str = Field(..., description="Country name, e.g. 'Albania'")
    Item: str = Field(..., description="Crop name, e.g. 'Maize'")
    Year: int = Field(..., ge=1990, le=2030, description="Year of record")
    average_rain_fall_mm_per_year: float = Field(
        ..., ge=0, le=5000, description="Annual rainfall in mm"
    )
    pesticides_tonnes: float = Field(
        ..., ge=0, le=400000, description="Pesticide use in tonnes"
    )
    avg_temp: float = Field(
        ..., ge=-10, le=50, description="Average temperature in Celsius"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "Area": "Albania",
                "Item": "Maize",
                "Year": 2010,
                "average_rain_fall_mm_per_year": 1485.0,
                "pesticides_tonnes": 121.0,
                "avg_temp": 16.37,
            }
        }


class PredictionOutput(BaseModel):
    predicted_yield_hg_per_ha: float


# ---------------------------------------------------------------------------
# Root + info endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Crop Yield Prediction API. See /docs for usage."}


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "n_features": len(feature_order),
        "feature_order": feature_order,
        "valid_areas_count": len(VALID_AREAS),
        "valid_items": VALID_ITEMS,
    }


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict(data: YieldInput):
    # Validate the categorical names against what the encoders know
    if data.Area not in VALID_AREAS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown Area '{data.Area}'. Must be one of the known countries.",
        )
    if data.Item not in VALID_ITEMS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown Item '{data.Item}'. Must be one of: {VALID_ITEMS}",
        )

    # Encode names to the same numbers used in training
    area_code = int(area_encoder.transform([data.Area])[0])
    item_code = int(item_encoder.transform([data.Item])[0])

    # Build a single-row frame in the exact feature order the model expects
    row = pd.DataFrame(
        [[
            area_code,
            item_code,
            data.Year,
            data.average_rain_fall_mm_per_year,
            data.pesticides_tonnes,
            data.avg_temp,
        ]],
        columns=feature_order,
    )

    # Scale with the SAME scaler, then predict
    row_scaled = scaler.transform(row)
    prediction = float(model.predict(row_scaled)[0])

    return PredictionOutput(predicted_yield_hg_per_ha=round(prediction, 2))


# ---------------------------------------------------------------------------
# Retrain endpoint: upload a CSV with the same columns to retrain the model
# ---------------------------------------------------------------------------
@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Please upload a .csv file.")

    contents = await file.read()
    try:
        new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {e}")

    # Drop a stray index column if present
    if "Unnamed: 0" in new_df.columns:
        new_df = new_df.drop(columns=["Unnamed: 0"])

    required = {"Area", "Item", "Year", "hg/ha_yield",
                "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"}
    if not required.issubset(new_df.columns):
        raise HTTPException(
            status_code=422,
            detail=f"CSV must contain columns: {sorted(required)}",
        )

    # Re-encode, re-split target, re-scale, retrain
    new_area_enc = LabelEncoder()
    new_item_enc = LabelEncoder()
    new_df = new_df.copy()
    new_df["Area"] = new_area_enc.fit_transform(new_df["Area"])
    new_df["Item"] = new_item_enc.fit_transform(new_df["Item"])

    X_new = new_df[feature_order]
    y_new = new_df["hg/ha_yield"]

    new_scaler = StandardScaler()
    X_new_scaled = new_scaler.fit_transform(X_new)

    new_model = RandomForestRegressor(n_estimators=100, random_state=42)
    new_model.fit(X_new_scaled, y_new)

    # Hot-swap the in-memory objects and persist to disk
    global model, scaler, area_encoder, item_encoder, VALID_AREAS, VALID_ITEMS
    model = new_model
    scaler = new_scaler
    area_encoder = new_area_enc
    item_encoder = new_item_enc
    VALID_AREAS = list(new_area_enc.classes_)
    VALID_ITEMS = list(new_item_enc.classes_)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "area_encoder": area_encoder,
            "item_encoder": item_encoder,
            "feature_order": feature_order,
        },
        "best_model.pkl",
    )

    return {
        "message": "Model retrained successfully on uploaded data.",
        "rows_used": len(new_df),
    }
