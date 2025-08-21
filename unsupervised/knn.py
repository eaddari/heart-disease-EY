import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('dataset\\heart_preprocessed.csv')

print(df.head())

X = df.drop('target', axis=1)

found_variables = df[['exang', 'cp', 'oldpeak', 'thalach', 'ca', 'slope', 'thal', 'sex']]

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

inertias = []
silhouette_scores = []
cluster_range = range(2, 11)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(found_variables)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(found_variables, kmeans.labels_))


final_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
final_kmeans.fit(found_variables)
df['cluster'] = final_kmeans.labels_
final_silhouette = silhouette_score(found_variables, final_kmeans.labels_); print(final_silhouette)

### PRINTS ###

plt.figure()
plt.plot(cluster_range, inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['exang'], df['cp'], df['oldpeak'], c=df['cluster'], cmap='viridis')
ax.set_xlabel('Exercise Induced Angina')
ax.set_ylabel('Chest Pain Type')
ax.set_zlabel('Oldpeak')
plt.show()
