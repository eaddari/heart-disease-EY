from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset\\heart_preprocessed.csv')

X = df.drop('target', axis=1)

agg_clustering = AgglomerativeClustering()

df['agg_cluster'] = agg_clustering.fit_predict(X)

links = linkage(X, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(links, labels=df['agg_cluster'].values, truncate_mode='lastp', show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

##### PCA POI CLUSTERING #####

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
print("Variabilita spiegata:", pca.explained_variance_ratio_)

agg_clustering2 = AgglomerativeClustering()
df['agg_cluster_2'] = agg_clustering2.fit_predict(X_pca)

plt.figure(figsize=(10, 7))
dendrogram(links, labels=df['agg_cluster_2'].values, truncate_mode='lastp', show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
