# Crop Yield Prediction for African Food Security

## Mission and Problem

Food security planning across Africa depends on knowing what farmland can realistically produce before a season ends. Yields vary widely between crops and countries, so planners cannot rely on a single average figure. This project predicts crop yield in hectograms per hectare for 31 African countries from climate and input conditions. The result is served through a public API and a mobile app so the estimate is available wherever it is needed.

## Dataset

Source: [Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset) on Kaggle, file `yield_df.csv`.

The dataset merges crop yield and pesticide figures from the FAO with rainfall and average temperature data from the World Bank. The full file covers 101 countries from 1990 to 2013. For this project it was filtered to Africa only and cleaned of 161 duplicate rows, leaving **5,362 records across 31 African countries and 10 staple crops** including maize, cassava, sorghum, yams and rice. Each row records one crop in one country in one year, alongside that country's annual rainfall, pesticide use and average temperature.

The six predictors are country, crop, year, rainfall (mm/year), pesticide use (tonnes) and average temperature (°C). The target is yield in hg/ha.

## API

**Base URL:** https://linear-regression-model-26zx.onrender.com

**Swagger UI:** https://linear-regression-model-26zx.onrender.com/docs

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Returns predicted yield for one set of inputs |
| `/retrain` | POST | Retrains the model on an uploaded CSV and serves it immediately |
| `/model-info` | GET | Reports the model type, feature order and valid countries and crops |

Example request:

```json
{
  "Area": "Rwanda",
  "Item": "Maize",
  "Year": 2010,
  "average_rain_fall_mm_per_year": 1200.0,
  "pesticides_tonnes": 95.0,
  "avg_temp": 19.5
}
```

Every input is type checked and range constrained through Pydantic. Rainfall must fall between 0 and 3000 mm, pesticides between 0 and 30000 tonnes, temperature between 5 and 40 °C, and year between 1990 and 2030. Country and crop names are validated against the categories the model was trained on. Out of range or unknown values return a 422 with a message naming the problem.

Note: the API runs on Render's free tier and sleeps after 15 minutes of inactivity. The first request after a period of inactivity may take up to a minute to respond.

## Model

Four models were compared on the same held out test set of 1,073 records:

| Model | RMSE | R² |
|---|---|---|
| SGD (Gradient Descent) | 64,912.34 | 0.0656 |
| Linear Regression | 64,842.14 | 0.0676 |
| **Decision Tree** | **14,456.28** | **0.9537** |
| Random Forest | 15,076.42 | 0.9496 |

The Decision Tree had the lowest loss and was saved as `best_model.pkl`. The linear models underfit badly because yield is driven mostly by which crop is being grown, a categorical relationship that a single straight line cannot capture. Full analysis is in `summative/linear_regression/multivariate.ipynb`.

## Video Demo

[YouTube link to be added]

## Repository Structure

```
linear_regression_model/
├── summative/
│   ├── linear_regression/
│   │   └── multivariate.ipynb      # EDA, model comparison, saved model
│   ├── API/
│   │   ├── prediction.py           # FastAPI service
│   │   ├── best_model.pkl          # Trained Decision Tree with scaler and encoders
│   │   └── requirements.txt
│   └── FlutterApp/                 # Mobile app
├── pyproject.toml
└── uv.lock
```

## Running the Mobile App

**Requirements:** Flutter SDK, Android Studio with an Android emulator or a physical Android device.

```bash
git clone https://github.com/kauriane1/Linear_regression_model.git
cd Linear_regression_model/summative/FlutterApp
flutter pub get
flutter run
```

The app opens on a single page with six input fields, pre-filled with a valid example. Enter a country, crop, year, rainfall, pesticide use and temperature, then tap **Predict**. The result appears below the button in hectograms per hectare and tonnes per hectare. Invalid or out of range values return an error message in the same area.

The API URL is already configured in `lib/api_service.dart`, so no additional setup is needed.

## Running the API Locally

```bash
cd summative/API
pip install -r requirements.txt
uvicorn prediction:app --reload
```

Then open http://127.0.0.1:8000/docs