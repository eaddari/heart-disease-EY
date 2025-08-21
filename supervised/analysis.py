import os
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "heart.csv")

df = pd.read_csv(DATASET_PATH)

numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

corr_matrix_df = df[numeric_features]

# # Calcola la matrice di correlazione
correlation_matrix = corr_matrix_df.corr()

# # Visualizza la matrice
print(correlation_matrix)

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

df["target_label"] = df["target"].map({1: "Diagnosi positiva", 0: "Problemi cardiaci"})

#UNIVARIATI
# Grafico univariato della variabile 'sex'
plt.figure(figsize=(5,4))
sns.countplot(x="sex", data=df, palette="Set2")

plt.xticks([0, 1], ["Femmine", "Maschi"])
plt.xlabel("Sesso")
plt.ylabel("Numero di pazienti")
plt.title("Distribuzione del sesso nel dataset")
plt.show()


#BIVARIATI
# --- GRAFICI CATEGORICHE ---
n_cat = len(categorical_features)
cols = 3
rows = math.ceil(n_cat / cols)

fig_cat, axes_cat = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), constrained_layout=True)
axes_cat = axes_cat.flatten() if n_cat > 1 else [axes_cat]

i = 0
for i, col in enumerate(categorical_features):
    ax = axes_cat[i]
    sns.countplot(data=df, x=col, hue="target_label", ax=ax)
    ax.set_title(f"{col} vs Diagnosi", fontsize=10)
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel("Conteggi", fontsize=8)
    ax.tick_params(labelsize=7)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", label_type="edge", padding=2, fontsize=5)
    if i == 0:  
        ax.legend(fontsize=7, title_fontsize=8)
    else:
        ax.get_legend().remove()  


for j in range(i + 1, rows * cols):
    fig_cat.delaxes(axes_cat[j])

plt.show()

# --- GRAFICI NUMERICHE ---
n_num = len(numeric_features)
cols = 3
rows = math.ceil(n_num / cols)

fig_num, axes_num = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), constrained_layout=True)
axes_num = axes_num.flatten() if n_num > 1 else [axes_num]

i = 0
for i, col in enumerate(numeric_features):
    ax = axes_num[i]
    sns.boxplot(data=df, x="target_label", y=col, ax=ax)
    ax.set_title(f"{col} vs Diagnosi", fontsize=10)
    ax.set_xlabel("Diagnosi", fontsize=8)
    ax.set_ylabel(col, fontsize=8)
    ax.tick_params(labelsize=7)
    
    if ax.get_legend() is not None:
        ax.legend(fontsize=7)


for j in range(i + 1, rows * cols):
    fig_num.delaxes(axes_num[j])

plt.show()

#Features che discriminano maggiormente --> fbs, cp, sex,exang, age, thalach, oldpeak, ca(numero arterie principali colorate dalla fluoroscopia)