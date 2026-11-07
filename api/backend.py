import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "training"))

from custom_model import ThresholdXGBClassifier

MODEL_PATH = BASE_DIR / "models" / "xgb_model2.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="financial-inclusion-machine-learning-model-API",
    description="API for serving machine learning predictions",
    version="1.0.0"
)


class PredictionRequest(BaseModel):

    country: str
    year: str
    location_type: str
    cellphone_access: str
    household_size: int
    age_of_respondent: int
    gender_of_respondent: str
    relationship_with_head: str
    marital_status: str
    education_level: str
    job_type: str

@app.get("/")
def home():
    return {"message": "ML Prediction API is running"}

@app.post("/predict")
def predict(data: PredictionRequest):

    try:

        # Convert incoming data into a dictionary
        input_data = data.model_dump()

        # Convert dictionary into DataFrame
        input_df = pd.DataFrame([input_data])

        # Generate prediction
        prediction = model.predict(input_df)

        return {
            "prediction": int(prediction[0])
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )






























