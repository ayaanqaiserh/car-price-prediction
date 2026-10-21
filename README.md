# Car Price Prediction

A machine learning project that predicts a car's selling price and price category using car details.

## Features

- Regression model predicts selling price.
- Classification model predicts a price category.
- FastAPI provides prediction endpoints.
- A web page provides separate forms for both predictions.

## Dataset

This project uses `car data.csv`, with details such as year, present price, kilometers driven, fuel type, seller type, transmission, and previous owners.

Keep the CSV file in the project folder before training the models.

## Price categories

The classification model uses these selling price ranges:

- Low: up to 2
- Medium: over 2 and up to 5
- High: over 5 and up to 10
- Very High: over 10

## Model results

- Regression R²: `0.963`
- Classification accuracy: `0.820`

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
` ` `