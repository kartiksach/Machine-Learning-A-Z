"""
Author @ kartiksach
Platform @ Kartik's Macbook Air

This file contains the entire K-Means Clustering code
as covered in Machine Learning A-Z
"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('KMeansClustering.csv')
X = dataset.iloc[:, [3, 4]].values

# Computing values of k = Number of clusters using elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
	kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
	kmeans.fit(X)
	wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#From the graph obtained of Elbow method, we take k = 5

# Predicting clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
y_pred = kmeans.fit_predict(X)

""" 
Visualising results
First attribute of plt.scatter() in line 45-49 matches the x cordinate(0) points in dataset which correspond to cluster i
Second attribute of plt.scatter() in line 45-49 matches the y cordinate(1) points in dataset which correspond to cluster i
First attribute of plt.scatter() in line 50 matches the x cordinate(0) of the centroids
Second attribute of plt.scatter() in line 50 matches the y cordinate(1) of the centroids
"""

plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], color = 'yellow', s = 20, label = 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], color = 'green', s = 20, label = 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], color = 'blue', s = 20, label = 'Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], color = 'cyan', s = 20, label = 'Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], color = 'magenta', s = 20, label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', s = 50, label = 'Centroid')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1 - 100')
plt.title('Clustering')
plt.show()