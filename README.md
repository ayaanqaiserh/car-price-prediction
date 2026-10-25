# Car Price Prediction

This project uses vehicle details to predict a selling price and a price category. It includes a regression model, a classification model, a FastAPI app, and a webpage with two prediction forms.

## Dataset

The project uses `car data.csv`. Keep this file in the project folder. The code uses year, present price, kilometers driven, fuel type, seller type, transmission, and previous owners.

## Run the project

Open a terminal in the project folder and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install pandas scikit-learn fastapi uvicorn joblib
```

Train and save the models:

```bash
python train_model.py
```

Start the app:

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` for the webpage or `http://127.0.0.1:8000/docs` for the API documentation. Keep the terminal open while using the app.

## Results

- Regression R²: `0.963`
- Classification accuracy: `0.820`

  
- The classification categories are Low (up to 2), Medium (over 2 to 5), High (over 5 to 10), and Very High (over 10), in the dataset's price units.
