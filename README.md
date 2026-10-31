# Car Price Prediction

Used car buyers and sellers need a fair price. This project predicts *how much a used car will sell for*, and which price group it falls into (Low, Medium, High or Very High).

- *Backend:* FastAPI (app.py)
- *Frontend:* HTML and JavaScript (index.html)
- *Dataset:* car data.csv (301 cars)

## How to run

In a terminal inside this folder, create the environment and install the libraries (first time only):


python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt


Train and save the models (this creates the models folder):


.venv/bin/python compare-models.py


Start the app:


.venv/bin/python -m uvicorn app:app --reload


Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

On Windows, use .venv/Scripts/python instead of .venv/bin/python. Run the training step before starting the app. The app will not start without the files in the models folder.

| Other commands | |
|---|---|
| Train all models again | .venv/bin/python compare-models.py |
| Run the first version (Random Forest only) | .venv/bin/python train_model.py |

## Dataset

- *Predict:* Selling_Price (a number), and Price_Category made from it: Low (up to 2), Medium (2 to 5), High (5 to 10), Very High (over 10).
- *Inputs:* year, present price, kilometers driven, fuel type, seller type, transmission, number of previous owners.
- *Not used as an input:* Selling_Price, because it is the value being predicted.
- *Class sizes:* Low 108, Medium 86, High 79, Very High 28. "Very High" is the rare class, so the split is stratified and precision, recall and F1 are checked, not only accuracy.

## Models

| Type | Model | Code |
|---|---|---|
| Regression | Simple Linear Regression (present price only) | compare-models.py |
| Regression | Multiple Linear Regression (7 inputs) | compare-models.py |
| Regression | Decision Tree | compare-models.py |
| Regression | Random Forest | compare-models.py |
| Regression | SVM (SVR) | compare-models.py |
| Classification | Decision Tree | compare-models.py |
| Classification | Random Forest | compare-models.py |
| Classification | SVM (SVC), tuned | compare-models.py |
| Classification | Naive Bayes (Gaussian and Multinomial) | compare-models.py |

Naive Bayes is used only for classification, because it is a probability classifier and does not predict numbers. In the web app, the "Linear Regression" option is the multiple version.

## Results

Tested on 61 cars the models did not train on (20% of the data, split by price category).

| Regression | R² |
|---|---|
| Simple Linear Regression | 0.783 |
| Multiple Linear Regression | 0.871 |
| Decision Tree | 0.880 |
| Random Forest | 0.921 |
| SVR | 0.941 |

| Classification | Accuracy |
|---|---|
| Multinomial Naive Bayes | 0.475 |
| Gaussian Naive Bayes | 0.770 |
| Decision Tree | 0.836 |
| Random Forest | 0.852 |
| SVM (RBF, C = 1) | 0.869 |
| SVM tuned (RBF, C = 100, gamma = 0.1) | 0.934 |

SVM tuning used 5-fold cross-validation on the training data. With C = 1 the linear kernel scored 0.829 and the RBF kernel 0.792. The best RBF setting scored 0.854 in cross-validation. The web app uses the tuned SVM. The test set is small (61 cars, only 6 in "Very High"), so differences between the top models can change with a different split.

## Folders


cars-price-prediction/
├── app.py              # backend: connects the web page to the models
├── index.html          # the web page
├── compare-models.py   # trains, compares and saves all models
├── train_model.py      # first version: Random Forest only
├── requirements.txt    # Python libraries needed
├── car data.csv        # the dataset
└── models/             # saved trained models (created by compare-models.py). 
