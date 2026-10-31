# Car Price Prediction

A small machine learning web app that predicts a used car's selling price (regression) and its price category (classification). It uses the CarDekho car dataset (car data.csv), scikit-learn for the models, FastAPI for the backend, and a plain HTML/JavaScript page for the frontend.

## What it predicts

- *Selling price:* a number (regression).
- *Price category:* Low (up to 2), Medium (2 to 5), High (5 to 10) or Very High (over 10), using the same price units as the dataset.

## Algorithms used


| Algorithm | Regression | Classification |
|---|---|---|
| Simple Linear Regression (1 input) | Yes | - |
| Multiple Linear Regression (7 inputs) | Yes | - |
| Decision Tree | Yes | Yes |
| Random Forest | Yes | Yes |
| SVM (SVR / SVC) | Yes | Yes |
| Naive Bayes (Gaussian, Multinomial) | - | Yes |

Naive Bayes is only used for classification, because it is a probability classifier and does not predict numbers. In the web app, the "Linear Regression" option is the multiple version.

## Results

Data split: 80% training (240 cars), 20% testing (61 cars), stratified by price category.

*Regression (R²)*

| Model | R² |
|---|---|
| Simple Linear Regression (present price only) | 0.783 |
| Multiple Linear Regression (7 inputs) | 0.871 |
| Decision Tree | 0.880 |
| Random Forest | 0.921 |
| SVR | 0.941 |

*Classification (accuracy)*

| Model | Accuracy |
|---|---|
| Multinomial Naive Bayes | 0.475 |
| Gaussian Naive Bayes | 0.770 |
| Decision Tree | 0.836 |
| Random Forest | 0.852 |
| SVM (RBF, C = 1) | 0.869 |
| SVM tuned (RBF, C = 100, gamma = 0.1) | 0.934 |

SVM tuning used 5-fold cross-validation on the training data. With C = 1 the linear kernel scored 0.829 and the RBF kernel 0.792. The best RBF setting (C = 100, gamma = 0.1) scored 0.854 in cross-validation. The web app uses the tuned SVM. The test set is small (61 cars, 6 in "Very High"), so differences between the top models can change with a different split.