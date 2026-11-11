import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException


from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

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

    model_config = ConfigDict(extra="forbid")

    country: Literal[
        "Rwanda",
        "Tanzania",
        "Kenya",
        "Uganda"
    ]

    year: Literal["2016", "2017", "2018"]

    location_type: Literal["Rural", "Urban"]

    cellphone_access: Literal["Yes", "No"]

    household_size: int = Field(
        ge=1,
        le=21,
        description="Household size must be between 1 and 21"
    )

    age_of_respondent: int = Field(
        ge=16,
        le=100,
        description="Age must be between 16 and 100"
    )

    gender_of_respondent: Literal["Female", "Male"]

    relationship_with_head: Literal[
    "Head of Household",
    "Spouse",
    "Child",
    "Parent",
    "Other relative",
    "Other non-relatives"
]

    marital_status: Literal[
    "Married/Living together",
    "Single/Never Married",
    "Widowed",
    "Divorced/Seperated",
    "Dont know"
]

    education_level: Literal[
    "Primary education",
    "Secondary education",
    "No formal education",
    "Tertiary education",
    "Vocational/Specialised training",
    "Other/Dont know/RTA"
]

    job_type: Literal[
    "Self employed",
    "Informally employed",
    "Farming and Fishing",
    "Remittance Dependent",
    "Formally employed Private",
    "Other Income",
    "No Income",
    "Formally employed Government",
    "Government Dependent",
    "Dont Know/Refuse to answer"
]

@app.get("/")
def home():
    return {"message": "ML Prediction API is running"}

@app.post("/predict")
def predict(data: PredictionRequest):

    try:
        # Convert incoming data into a DataFrame
        input_data_dict = data.model_dump()
        input_data = pd.DataFrame([input_data_dict])

        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]

        # Extract probabilities for both classes
        no_account_probability = float(probabilities[0])
        account_probability = float(probabilities[1])

        # Get prediction using your custom 0.40 threshold
        prediction = int(model.predict(input_data)[0])

        # Generate prediction label
        prediction_label = (
            "Has a bank account"
            if prediction == 1
            else "Does not have a bank account"
        )

        return {
            "prediction": prediction,
            "prediction_label": prediction_label,

            "probabilities": {
                "has_bank_account": round(account_probability, 4),
                "no_bank_account": round(no_account_probability, 4)
            },

            "classification_threshold": 0.40
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )





























