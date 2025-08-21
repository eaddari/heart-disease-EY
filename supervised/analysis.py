import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from supervised import DATASET_PATH

df = pd.read_csv(DATASET_PATH)

corr_matrix_df = df[["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]]

# # Calcola la matrice di correlazione
correlation_matrix = corr_matrix_df.corr()

# # Visualizza la matrice
print(correlation_matrix)

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()


# # #### Selezione delle feature
# X = df[["hour", "dayofweek", "month"]]
# y = df["target"]

# # #### Split del dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, stratify=y, test_size=0.25, random_state=42
# )
