from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
from fastapi.responses import FileResponse

app = FastAPI(title="Car Price Prediction API")

regression_model = load("regression_model.joblib")
classification_model = load("classification_model.joblib")


class CarDetails(BaseModel):
    Year: int
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: str
    Seller_Type: str
    Transmission: str
    Owner: int


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
    price = regression_model.predict(make_input(car))[0]
    return {"predicted_selling_price": round(float(price), 2)}


@app.post("/predict/classification")
def predict_classification(car: CarDetails):
    category = classification_model.predict(make_input(car))[0]
    return {"predicted_price_category": str(category)}
@app.get("/")
def home():
    return FileResponse("index.html")