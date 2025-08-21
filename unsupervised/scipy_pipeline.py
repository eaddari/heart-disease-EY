import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

df = pd.read_csv('dataset\\heart_preprocessed.csv')
X = df.drop('target', axis=1).values

X_whitened = whiten(X)

centroids, labels = kmeans2(X_whitened, k=5, minit='points')

linkage_matrix = linkage(X_whitened, method='ward')
hier_labels = fcluster(linkage_matrix, t=5, criterion='maxclust')

df['kmeans_cluster'] = labels
df['hier_cluster'] = hier_labels - 1

print(f"K-means clusters: {np.bincount(labels)}")
print(f"Hierarchical clusters: {np.bincount(hier_labels - 1)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(df['age'], df['chol'], c=df['kmeans_cluster'], cmap='viridis')
ax1.set_title('K-means (SciPy)')
ax2.scatter(df['age'], df['chol'], c=df['hier_cluster'], cmap='viridis')
ax2.set_title('Hierarchical (SciPy)')
plt.show()
