# Car Price Prediction

## Project Overview

Car Price Prediction is a full-stack machine learning application. It predicts a vehicle's selling price and classifies the price as Low, Medium, High, or Very High using vehicle details.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- FastAPI
- HTML and JavaScript
- Joblib

## Machine Learning Models

- **Regression:** Random Forest Regressor predicts the selling-price number.
- **Classification:** Random Forest Classifier predicts the price category.
- Text details such as fuel type are converted into a format the models can use with one-hot encoding.

### Model Evaluation

The models were evaluated using a held-out test set.

| Metric | Result |
|---|---:|
| Regression R² Score | 0.963 |
| Classification Accuracy | 82.0% |

The price categories use the dataset's price units: Low is up to 2, Medium is over 2 and up to 5, High is over 5 and up to 10, and Very High is over 10.

## Project Structure

The main project files are in the project folder:

- `app.py` — FastAPI application and prediction endpoints
- `train_model.py` — prepares the data, trains the models, prints the scores, and saves the models
- `index.html` — webpage with separate regression and classification forms
- `car data.csv` — dataset used by the training script
- `regression_model.joblib` — saved regression model and preprocessing steps
- `classification_model.joblib` — saved classification model and preprocessing steps
- `README.md` — project information and run instructions

The repository also contains other downloaded CSV files. The training script uses `car data.csv`.

## How to Run the Project on macOS

Open a terminal in the project folder.

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the required packages

```bash
python -m pip install pandas scikit-learn fastapi uvicorn joblib
```

### 3. Train and save the models

Make sure `car data.csv` is in the project folder, then run:

```bash
python train_model.py
```

This prints the evaluation scores and creates the two `.joblib` model files.

### 4. Start the application

```bash
uvicorn app:app --reload
```

Keep the terminal open while using the app. Open `http://127.0.0.1:8000` for the webpage or `http://127.0.0.1:8000/docs` to view and try the API. The GitHub repository stores the project files; the app runs locally using these steps.

## API Endpoints

- `POST /predict/regression` — predicts a selling price.
- `POST /predict/classification` — predicts a price category.
