"""Crop yield prediction API for African food security.

Serves a Decision Tree model trained on FAO and World Bank data
covering 31 African countries between 1990 and 2013.
"""

import io

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

bundle = joblib.load("best_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
area_encoder = bundle["area_encoder"]
item_encoder = bundle["item_encoder"]
feature_order = bundle["feature_order"]

VALID_AREAS = list(area_encoder.classes_)
VALID_ITEMS = list(item_encoder.classes_)

app = FastAPI(
    title="Crop Yield Prediction API",
    description="Predicts crop yield (hg/ha) across African countries to support food security planning.",
    version="1.0.0",
)

# Origins are listed explicitly rather than using a wildcard. Only the local
# dev hosts and the deployed service need access. Methods are limited to GET
# and POST since nothing else is exposed, and only Content-Type is allowed
# because requests carry plain JSON. Credentials are off - there are no
# cookies, sessions or auth tokens to send.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://linear-regression-model-26zx.onrender.com",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class YieldInput(BaseModel):
    """Bounds come from the African data, with a little headroom.

    Observed ranges: rainfall 51-2041 mm, pesticides 0.04-26857 t,
    temperature 13.9-30.7 C.
    """

    Area: str = Field(..., description="African country name, e.g. 'Rwanda'")
    Item: str = Field(..., description="Crop name, e.g. 'Maize'")
    Year: int = Field(..., ge=1990, le=2030)
    average_rain_fall_mm_per_year: float = Field(..., ge=0, le=3000)
    pesticides_tonnes: float = Field(..., ge=0, le=30000)
    avg_temp: float = Field(..., ge=5, le=40, description="Degrees Celsius")

    class Config:
        json_schema_extra = {
            "example": {
                "Area": "Rwanda",
                "Item": "Maize",
                "Year": 2010,
                "average_rain_fall_mm_per_year": 1200.0,
                "pesticides_tonnes": 95.0,
                "avg_temp": 19.5,
            }
        }


class PredictionOutput(BaseModel):
    predicted_yield_hg_per_ha: float


@app.get("/")
def root():
    return {"message": "Crop yield prediction API for African food security. See /docs."}


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "n_features": len(feature_order),
        "feature_order": feature_order,
        "valid_areas": VALID_AREAS,
        "valid_items": VALID_ITEMS,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: YieldInput):
    if data.Area not in VALID_AREAS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown Area '{data.Area}'. Must be one of: {VALID_AREAS}",
        )
    if data.Item not in VALID_ITEMS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown Item '{data.Item}'. Must be one of: {VALID_ITEMS}",
        )

    area_code = int(area_encoder.transform([data.Area])[0])
    item_code = int(item_encoder.transform([data.Item])[0])

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

    # Same scaler that was fitted during training, so the values mean the same thing
    prediction = float(model.predict(scaler.transform(row))[0])
    return PredictionOutput(predicted_yield_hg_per_ha=round(prediction, 2))


@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    """Retrain on an uploaded CSV and start serving the new model straight away."""
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Please upload a .csv file.")

    contents = await file.read()
    try:
        new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {e}")

    if "Unnamed: 0" in new_df.columns:
        new_df = new_df.drop(columns=["Unnamed: 0"])

    required = {
        "Area",
        "Item",
        "Year",
        "hg/ha_yield",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    }
    missing = required - set(new_df.columns)
    if missing:
        raise HTTPException(
            status_code=422, detail=f"CSV is missing columns: {sorted(missing)}"
        )

    new_df = new_df.drop_duplicates()
    if len(new_df) < 50:
        raise HTTPException(
            status_code=422,
            detail="Need at least 50 rows after removing duplicates to retrain.",
        )

    new_area_encoder = LabelEncoder()
    new_item_encoder = LabelEncoder()
    new_df["Area"] = new_area_encoder.fit_transform(new_df["Area"])
    new_df["Item"] = new_item_encoder.fit_transform(new_df["Item"])

    X = new_df[feature_order]
    y = new_df["hg/ha_yield"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    new_scaler = StandardScaler()
    X_train_scaled = new_scaler.fit_transform(X_train)
    X_test_scaled = new_scaler.transform(X_test)

    new_model = DecisionTreeRegressor(random_state=42)
    new_model.fit(X_train_scaled, y_train)

    preds = new_model.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    global model, scaler, area_encoder, item_encoder, VALID_AREAS, VALID_ITEMS
    model = new_model
    scaler = new_scaler
    area_encoder = new_area_encoder
    item_encoder = new_item_encoder
    VALID_AREAS = list(new_area_encoder.classes_)
    VALID_ITEMS = list(new_item_encoder.classes_)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "area_encoder": area_encoder,
            "item_encoder": item_encoder,
            "feature_order": feature_order,
        },
        "best_model.pkl",
        compress=3,
    )

    return {
        "message": "Model retrained and now serving predictions.",
        "rows_used": len(new_df),
        "test_rmse": round(rmse, 2),
        "test_r2": round(r2, 4),
    }