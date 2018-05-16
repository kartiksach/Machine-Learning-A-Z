"""
Author @ kartiksach
Platform @ Kartik's Macbook Air

This file contains the entire Heirarchical Clustering code
as covered in Machine Learning A-Z
"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('HierarchicalClustering.csv')
X = dataset.iloc[:, [3, 4]].values

""" 
Constructing Dendrogram to determine number of clusters
Linkage method is used to determine the method used to form clusters
Ward attribue of linkage forms clusters on basis of variance of distance of datapoints
"""
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

"""
From the dendrogram plot obtained, the longest vertical line
which doesnt interfere with any horizontal line is that with
euclidean distance from 100 to ~240. We draw a threshold value at 
that line. Now, 5 vertical lines cross the horizontal threshold.
Hence, number of clusters = 5
"""

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_pred = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
