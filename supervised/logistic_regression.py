import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV


DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "heart_preprocessed.csv")

df = pd.read_csv(DATASET_PATH)
X = df.drop("target", axis=1)
y = df["target"]
print(df.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

## LOGISTIC REGRESSION ###
#TUTTE LE VARIABILI

log_model = LogisticRegression(max_iter=1000, random_state=42)

log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

class_report = classification_report(y_test, y_pred)
print(class_report)

param_grid = {"C": [0.1, 1, 10], "solver": ["liblinear", "saga"]}

grid = GridSearchCV(
    estimator=log_model, param_grid=param_grid, scoring="accuracy", cv=5
)

grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)

# ### LOGISTIC REGRESSION ###
# ##FEATURE SELECTION
# X = df[['exang', 'cp', 'oldpeak', 'thalach', 'ca', 'slope', 'thal', 'sex']]
# y = df["target"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )


# log_model = LogisticRegression(max_iter=1000, random_state=42)

# log_model.fit(X_train, y_train)

# y_pred = log_model.predict(X_test)

# class_report = classification_report(y_test, y_pred)
# print(class_report)

# param_grid = {"C": [0.1, 1, 10], "solver": ["liblinear", "saga"]}

# grid = GridSearchCV(
#     estimator=log_model, param_grid=param_grid, scoring="accuracy", cv=5
# )

# grid.fit(X_train, y_train)

# print("Best parameters:", grid.best_params_)
# print("Best cross-validation score:", grid.best_score_)
