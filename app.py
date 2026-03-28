from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class CarFeatures(BaseModel):
    symboling: float
    wheelbase: float
    carlength: float
    carwidth: float
    curbweight: float
    enginesize: float
    stroke: float
    horsepower: float
    citympg: float
    body_type: str
    drive_wheel: str
    cylinders: str
    fuel_sys: str
    engine_type: str

try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('ridge_model.pkl')
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

@app.get("/")
def read_root():
    return {
        "message": "Car Price Predictor API is running. POST to /predict with JSON payload."
    }

@app.post("/predict")
def predict(data: CarFeatures):
    input_data = {feat: 0.0 for feat in scaler.feature_names_in_}
    input_data.update({
        'symboling': data.symboling,
        'wheelbase': data.wheelbase,
        'carlength': data.carlength,
        'carwidth': data.carwidth,
        'curbweight': data.curbweight,
        'enginesize': data.enginesize,
        'horsepower': data.horsepower,
        'citympg': data.citympg,
        'stroke': data.stroke,
    })

    for field_name, value in {
        'carbody_': data.body_type.lower(),
        'drivewheel_': data.drive_wheel,
        'fuelsystem_': data.fuel_sys,
        'enginetype_': data.engine_type,
        'cylindernumber_': data.cylinders,
    }.items():
        key = field_name + value
        if key in input_data:
            input_data[key] = 1.0

    features_array = np.array([list(input_data.values())])

    try:
        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"estimated_price": float(prediction[0])}
