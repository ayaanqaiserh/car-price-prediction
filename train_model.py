import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from joblib import dump


data = pd.read_csv("car data.csv")

data = data[
    [
        "Year",
        "Selling_Price",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner",
    ]
].dropna()


data["Price_Category"] = pd.cut(
    data["Selling_Price"],
    bins=[-float("inf"), 2, 5, 10, float("inf")],
    labels=["Low", "Medium", "High", "Very High"],
)

features = [
    "Year",
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
]
categorical_features = ["Fuel_Type", "Seller_Type", "Transmission"]

X = data[features]
y_regression = data["Selling_Price"]
y_classification = data["Price_Category"]


X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = (
    train_test_split(
        X,
        y_regression,
        y_classification,
        test_size=0.2,
        random_state=42,
    )
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough",
)

regression_model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
    ]
)

classification_model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]
)

regression_model.fit(X_train, y_reg_train)
classification_model.fit(X_train, y_cls_train)

regression_predictions = regression_model.predict(X_test)
classification_predictions = classification_model.predict(X_test)

print(f"Regression R²: {r2_score(y_reg_test, regression_predictions):.3f}")
print(
    "Classification accuracy: "
    f"{accuracy_score(y_cls_test, classification_predictions):.3f}"
)

dump(regression_model, "regression_model.joblib")
dump(classification_model, "classification_model.joblib")

print("Saved regression_model.joblib and classification_model.joblib")