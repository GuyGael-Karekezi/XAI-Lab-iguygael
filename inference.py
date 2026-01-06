from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import config
import pandas as pd
import time
import os

# Wait for model to exist
while not os.path.exists(config.MODEL_PATH):
    print(f"Waiting for model {config.MODEL_PATH}...")
    time.sleep(5)  # wait 5 seconds

app = FastAPI(title="Credit Risk Prediction API")

# Load the trained pipeline
model = joblib.load(config.MODEL_PATH)


class CreditInput(BaseModel):
    Age: int
    Sex: str
    Job: int
    Housing: str
    Saving_accounts: str
    Checking_account: float
    Credit_amount: float
    Duration: int
    Purpose: str


@app.post("/predict")
def predict_credit_risk(data: CreditInput):
    # Convert input to DataFrame with exact training column names
    df = pd.DataFrame([{
        "Age": data.Age,
        "Sex": data.Sex,
        "Job": data.Job,
        "Housing": data.Housing,
        "Saving accounts": data.Saving_accounts,
        "Checking account": data.Checking_account,
        "Credit amount": data.Credit_amount,
        "Duration": data.Duration,
        "Purpose": data.Purpose,
    }])

    # Prediction
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0, 1]

    return {
        "Risk": "Good" if pred == 1 else "Bad",
        "Probability": float(prob),
    }
