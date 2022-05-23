# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 13:50:50 2021

@author: HRollins
"""

import mglearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

#%%
mglearn.plots.plot_agglomerative_algorithm()

# %%

from sklearn.cluster import AgglomerativeClustering
X, y = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.legend(["Cluster 0", "Cluster 1", "Cluster 2"], loc="best")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# In[63]:

mglearn.plots.plot_agglomerative()

# In[64]: Hierarchical clustering without a fixed number of clusters

from sklearn.cluster import AgglomerativeClustering

X, y = make_blobs(random_state=6, n_samples=[50, 50, 50, 50, 50])

#first, we can set the threshold=0 to see all clusters.
#when distance_threshold is set, n_clusters must be None
#the higher the threshold, the fewer clusters we have
agg = AgglomerativeClustering(distance_threshold=15, n_clusters=None).fit(X)

#this gives use the distance between clusters in increasing order.
#the first element shows distance between cluster one and its closest cluster.
dist = agg.distances_ #we look at the distances to determine best threshold

pred = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], pred)
print(np.unique(pred))

# In[65]: Failure case for Hierarchical Clustering

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

agg = AgglomerativeClustering(n_clusters=5)

pred = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], pred)

# In[66]: Visualize the hierarchy of clusters
    
#Import the dendrogram function and the ward clustering function from SciPy
from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(random_state=0, n_samples=12)

# Apply the ward clustering to the data array X
# The SciPy ward function returns an array that specifies the distances
# bridged when performing agglomerative clustering
linkage_array = ward(X)

# Now we plot the dendrogram for the linkage_array containing the distances
# between clusters
dendrogram(linkage_array)

# mark the cuts in the tree that signify two or three clusters
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")
