#!/usr/bin/env python
# coding: utf-8

# 

# In[2]:


import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data generation
# The function below will generate the data points and requires these inputs:
# 
# centroidLocation: Coordinates of the centroids that will generate the random data.
# Example: input: [[4,3], [2,-1], [-1,4]]
# numSamples: The number of data points we want generated, split over the number of centroids (# of centroids defined in centroidLocation)
# Example: 1500
# clusterDeviation: The standard deviation of the clusters. The larger the number, the further the spacing of the data points within the clusters.
# Example: 0.5

# In[3]:


def createdatapoints(centroidLocation, numSamples, clusterDeviation):
    x, y=make_blobs(n_samples=numSamples, centers=centroidLocation, cluster_std=clusterDeviation)
    x=StandardScaler().fit_transform(x)
    return x, y


# In[4]:


x,y =createdatapoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)
print(x)
print(y)


# # Modeling
# DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. This technique is one of the most common clustering algorithms which works based on density of object. The whole idea is that if a particular point belongs to a cluster, it should be near to lots of other points in that cluster.
# 
# It works based on two parameters: # Epsilon and Minimum Points
# Epsilon determine a specified radius that if includes enough number of points within, we call it dense area
# # minimumSamples determine the minimum number of data points we want in a neighborhood to define a cluster.

# In[6]:


epsilon= 0.3
minimumsamples= 7
db= DBSCAN(eps= epsilon, min_samples=minimumsamples ).fit(x)
labels=db.labels_
labels


# In[7]:


# Firts, create an array of booleans using the labels from db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask


# In[8]:


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_


# In[9]:


# Remove repetition in labels by turning it into a set.
unique_labels = set(labels)
unique_labels


# # Data visualization

# In[10]:


# Create colors for the clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))


# In[15]:


# Plot the points with colors
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    # Plot the datapoints that are clustered
    xy = x[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

    # Plot the outliers
    xy = x[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)


# In[ ]:




