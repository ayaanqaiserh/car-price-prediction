# Car Price Prediction

A small machine learning web app that predicts a used car's selling price (regression) and its price category (classification). It uses the CarDekho car dataset (car data.csv), scikit-learn for the models, FastAPI for the backend, and a plain HTML/JavaScript page for the frontend.

## What it predicts

- *Selling price:* a number (regression).
- *Price category:* Low (up to 2), Medium (2 to 5), High (5 to 10) or Very High (over 10), using the same price units as the dataset.

## Algorithms used

| Algorithm | Regression | Classification |
|---|---|---|
| Linear Regression | Yes | - |
| Decision Tree | Yes | Yes |
| Random Forest | Yes | Yes |
| SVM (SVR / SVC) | Yes | Yes |
| Naive Bayes | - | Yes |

Naive Bayes is only used for classification, because it is a probability classifier and does not predict numbers.

## Results

Data split: 80% training (240 cars), 20% testing (61 cars), stratified by price category.

*Regression (R²)*

| Model | R² |
|---|---|
| Linear Regression | 0.871 |
| Decision Tree | 0.880 |
| Random Forest | 0.921 |
| SVR | 0.941 |

*Classification (accuracy)*

| Model | Accuracy |
|---|---|
| Naive Bayes | 0.770 |
| Decision Tree | 0.836 |
| Random Forest | 0.852 |
| SVM | 0.869 |

The test set is small (61 cars, only 6 in the "Very High" category), so the differences between the top models are small and can change with a different split.

## Project files

- car data.csv: the dataset.
- train_model.py: the first version, trains Random Forest only.
- compare-models.py: trains and compares all models, prints the scores, and saves them in the models folder.
- app.py: the FastAPI backend.
- index.html: the frontend page.
- requirements.txt: the Python libraries needed.

## How to run it

1. Open a terminal in the project folder.
2. Create and activate a virtual environment:
   
   python3 -m venv .venv
   source .venv/bin/activate
   
3. Install the libraries:
   
   pip install -r requirements.txt
   
4. Train and save the models (this creates the models folder):
   
   python compare-models.py
   
5. Start the server:
   
   uvicorn app:app --reload
   
6. Open http://127.0.0.1:8000 in your browser.

Always start the server from inside the project folder, or the model files will not be found.

## API endpoints

Both endpoints accept a POST request with JSON like this:

json
{
  "Year": 2014,
  "Present_Price": 5.59,
  "Kms_Driven": 27000,
  "Fuel_Type": "Petrol",
  "Seller_Type": "Dealer",
  "Transmission": "Manual",
  "Owner": 0,
  "algorithm": "random_forest"
}


- POST /predict/regression returns {"predicted_selling_price": ..., "algorithm": ...}
- POST /predict/classification returns {"predicted_price_category": ..., "algorithm": ...}

Valid algorithm values: linear_regression, decision_tree, random_forest, svm for regression, and decision_tree, random_forest, svm, naive_bayes for classification. If algorithm is left out, random_forest is used.
