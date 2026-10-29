import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("car data.csv")

data["Price_Category"] = pd.cut(
    data["Selling_Price"],
    bins=[-float("inf"), 2, 5, 10, float("inf")],
    labels=["Low", "Medium", "High", "Very High"],
)

features = ["Year", "Present_Price", "Kms_Driven",
            "Fuel_Type", "Seller_Type", "Transmission", "Owner"]

X = data[features]
y_price = data["Selling_Price"]
y_category = data["Price_Category"]

X_train, X_test, y_price_train, y_price_test, y_cat_train, y_cat_test = (
    train_test_split(X, y_price, y_category,
                     test_size=0.2, random_state=42, stratify=y_category)
)

print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print(y_cat_test.value_counts())


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score, classification_report

categorical = ["Fuel_Type", "Seller_Type", "Transmission"]

def make_preprocessor():
    return ColumnTransformer(
        [("cats", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )

# Decision Tree for regression
tree_reg = Pipeline([
    ("prep", make_preprocessor()),
    ("model", DecisionTreeRegressor(max_depth=5, random_state=42)),
])
tree_reg.fit(X_train, y_price_train)
print("Decision Tree R2:", round(r2_score(y_price_test, tree_reg.predict(X_test)), 3))

# Decision Tree for classification
tree_cls = Pipeline([
    ("prep", make_preprocessor()),
    ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
])
tree_cls.fit(X_train, y_cat_train)
tree_pred = tree_cls.predict(X_test)
print("Decision Tree accuracy:", round(accuracy_score(y_cat_test, tree_pred), 3))
print(classification_report(y_cat_test, tree_pred, zero_division=0))

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Random Forest for regression
rf_reg = Pipeline([
    ("prep", make_preprocessor()),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
])
rf_reg.fit(X_train, y_price_train)
print("Random Forest R2:", round(r2_score(y_price_test, rf_reg.predict(X_test)), 3))

# Random Forest for classification
rf_cls = Pipeline([
    ("prep", make_preprocessor()),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
])
rf_cls.fit(X_train, y_cat_train)
rf_pred = rf_cls.predict(X_test)
print("Random Forest accuracy:", round(accuracy_score(y_cat_test, rf_pred), 3))
print(classification_report(y_cat_test, rf_pred, zero_division=0))


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC

numeric = ["Year", "Present_Price", "Kms_Driven", "Owner"]

def make_scaled_preprocessor():
    return ColumnTransformer([
        ("cats", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("nums", StandardScaler(), numeric),
    ])

# SVM for regression
svr = Pipeline([
    ("prep", make_scaled_preprocessor()),
    ("model", SVR(kernel="rbf", C=100)),
])
svr.fit(X_train, y_price_train)
print("SVR R2:", round(r2_score(y_price_test, svr.predict(X_test)), 3))

# SVM for classification
svc = Pipeline([
    ("prep", make_scaled_preprocessor()),
    ("model", SVC(kernel="rbf", C=1)),
])
svc.fit(X_train, y_cat_train)
svc_pred = svc.predict(X_test)
print("SVM accuracy:", round(accuracy_score(y_cat_test, svc_pred), 3))
print(classification_report(y_cat_test, svc_pred, zero_division=0))

from sklearn.naive_bayes import GaussianNB

nb = Pipeline([
    ("prep", ColumnTransformer(
        [("cats", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)],
        remainder="passthrough",
    )),
    ("model", GaussianNB()),
])
nb.fit(X_train, y_cat_train)
nb_pred = nb.predict(X_test)
print("Naive Bayes accuracy:", round(accuracy_score(y_cat_test, nb_pred), 3))
print(classification_report(y_cat_test, nb_pred, zero_division=0))


from sklearn.linear_model import LinearRegression

lin_reg = Pipeline([
    ("prep", make_scaled_preprocessor()),
    ("model", LinearRegression()),
])
lin_reg.fit(X_train, y_price_train)
print("Linear Regression R2:", round(r2_score(y_price_test, lin_reg.predict(X_test)), 3))

import os
from joblib import dump

os.makedirs("models", exist_ok=True)

regression_models = {
    "linear_regression": lin_reg,
    "decision_tree": tree_reg,
    "random_forest": rf_reg,
    "svm": svr,
}
classification_models = {
    "decision_tree": tree_cls,
    "random_forest": rf_cls,
    "svm": svc,
    "naive_bayes": nb,
}

for name, model in regression_models.items():
    dump(model, f"models/regression_{name}.joblib")
for name, model in classification_models.items():
    dump(model, f"models/classification_{name}.joblib")

print("Saved", len(regression_models) + len(classification_models), "models in the models folder")