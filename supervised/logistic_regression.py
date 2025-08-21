import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from supervised import DATASET_PATH


df = pd.read_csv(DATASET_PATH)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

### LOGISTIC REGRESSION ###

log_model = LogisticRegression()

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
