from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from joblib import load
import pandas as pd

app = FastAPI(title="Car Price Prediction API")

regression_models = {
    "linear_regression": load("models/regression_linear_regression.joblib"),
    "decision_tree": load("models/regression_decision_tree.joblib"),
    "random_forest": load("models/regression_random_forest.joblib"),
    "svm": load("models/regression_svm.joblib"),
}
classification_models = {
    "decision_tree": load("models/classification_decision_tree.joblib"),
    "random_forest": load("models/classification_random_forest.joblib"),
    "svm": load("models/classification_svm.joblib"),
    "naive_bayes": load("models/classification_naive_bayes.joblib"),
}


class CarDetails(BaseModel):
    Year: int
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: str
    Seller_Type: str
    Transmission: str
    Owner: int
    algorithm: str = "random_forest"


def make_input(car: CarDetails):
    return pd.DataFrame([{
        "Year": car.Year,
        "Present_Price": car.Present_Price,
        "Kms_Driven": car.Kms_Driven,
        "Fuel_Type": car.Fuel_Type,
        "Seller_Type": car.Seller_Type,
        "Transmission": car.Transmission,
        "Owner": car.Owner,
    }])


@app.post("/predict/regression")
def predict_regression(car: CarDetails):
    if car.algorithm not in regression_models:
        raise HTTPException(status_code=400, detail="Unknown algorithm for regression.")
    model = regression_models[car.algorithm]
    price = model.predict(make_input(car))[0]
    return {"predicted_selling_price": round(float(price), 2),
            "algorithm": car.algorithm}


@app.post("/predict/classification")
def predict_classification(car: CarDetails):
    if car.algorithm not in classification_models:
        raise HTTPException(status_code=400, detail="Unknown algorithm for classification.")
    model = classification_models[car.algorithm]
    category = model.predict(make_input(car))[0]
    return {"predicted_price_category": str(category),
            "algorithm": car.algorithm}


@app.get("/")
def home():
    return FileResponse("index.html")

