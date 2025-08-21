import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "heart.csv")

df = pd.read_csv(DATASET_PATH)


X = df.drop('target', axis=1)
y = df['target']
print(df.shape)


feature_importance = abs(X.corrwith(pd.Series(y))).sort_values(ascending=False)
ordered_features = feature_importance.index.tolist()

# Valuta il modello aggiungendo una feature alla volta
results = []
for i in range(1, len(ordered_features) + 1):
    # Seleziona le prime i feature
    selected = ordered_features[:i]
    
    # Valuta il modello
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X[selected], y, cv=5, scoring='accuracy')
    
    results.append({
        'num_features': i,
        'features': selected,
        'accuracy': np.mean(scores),
        'std_dev': np.std(scores)
    })

# Visualizza i risultati
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='accuracy', ascending=False)
# Print features of best model
print(results_df.iloc[0, [1]].tolist())

# # Grafico delle prestazioni in funzione del numero di features
# plt.figure(figsize=(10, 6))
# plt.errorbar(results_df['num_features'], results_df['accuracy'], 
#             yerr=results_df['std_dev'], marker='o')
# plt.xlabel('Numero di features')
# plt.ylabel('Accuracy (cross-validation)')
# plt.title('Prestazioni del modello al variare del numero di features')
# plt.xticks(range(1, len(ordered_features) + 1))
# plt.grid(True)
# plt.tight_layout()
# plt.show()