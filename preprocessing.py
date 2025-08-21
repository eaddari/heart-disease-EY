import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('21_08/dataset/heart.csv')

# Analisi statistica delle variabili
print(df.describe())

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print(df.describe())

df.to_csv('21_08/dataset/heart_preprocessed.csv', index=False)
