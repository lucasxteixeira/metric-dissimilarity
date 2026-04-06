"""Prototype selection via clustering.

Provides methods to compute class-level prototypes from embeddings
using K-Means, Spectral, or Agglomerative clustering.
"""

import os
import pickle

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering


def compute_centroids(X, Y, K):
  """
  Computes centroids for each cluster.

  Parameters
  ----------
  X : np.ndarray
      The input data of shape (num_samples, num_features).
  Y : np.ndarray
      The cluster labels corresponding to each sample.
  K : int
      The number of clusters.

  Returns
  -------
  np.ndarray
      A NumPy array of shape (K, num_features) containing the centroids.
  """

  m, n = X.shape
  centroids = np.zeros((K, n))
  for k in range(K):
    x = X[Y == k]
    centroids[k, :] = np.mean(x, axis=0)
  return centroids


def compute_prototypes(embeddings, Y, n_prototypes=5, method="kmeans++", cache="prototypes.pkl"):
  """
  Computes prototypes for each class using clustering.

  Parameters
  ----------
  embeddings : np.ndarray
      The embeddings data of shape (num_samples, num_features).
  Y : np.ndarray
      The class labels corresponding to each sample.
  n_prototypes : int, optional
      The number of prototypes per class. Defaults to 5.
  method : str, optional
      Clustering method: 'kmeans', 'kmeans++', 'spectral', or 'hierarchical'. Defaults to 'kmeans++'.
  cache : str or False, optional
      File path for caching prototypes. Set to False to disable caching. Defaults to 'prototypes.pkl'.

  Returns
  -------
  np.ndarray
      Prototypes of shape (total_prototypes, num_features).
  np.ndarray
      Class label for each prototype of shape (total_prototypes,).
  """

  if cache is not False and os.path.isfile(cache):
    with open(cache, "rb") as f:
      prototypes, classes = pickle.load(f)
    return prototypes, classes

  uniq_classes = np.unique(Y)
  n_classes = len(uniq_classes)

  total_prototypes = n_classes * n_prototypes
  prototypes = np.zeros((total_prototypes, embeddings.shape[1]), dtype=np.float32)
  classes = np.zeros(total_prototypes, dtype=np.int32)

  for idx, cls in enumerate(uniq_classes):
    X_embedding = embeddings[np.where(Y == cls)]
    start, end = n_prototypes * (idx + 1) - n_prototypes, n_prototypes * (idx + 1)

    if method == "kmeans":
      clustering = KMeans(n_clusters=n_prototypes, init="random", n_init="auto", random_state=1234).fit(X_embedding)
      centroids = clustering.cluster_centers_
    elif method == "kmeans++":
      clustering = KMeans(n_clusters=n_prototypes, init="k-means++", n_init="auto", random_state=1234).fit(X_embedding)
      centroids = clustering.cluster_centers_
    elif method == "spectral":
      clustering = SpectralClustering(n_clusters=n_prototypes, affinity="nearest_neighbors", random_state=1234).fit(X_embedding)
      labels = clustering.labels_
      centroids = compute_centroids(X_embedding, labels, n_prototypes)
    elif method == "hierarchical":
      clustering = AgglomerativeClustering(n_clusters=n_prototypes).fit(X_embedding)
      labels = clustering.labels_
      centroids = compute_centroids(X_embedding, labels, n_prototypes)
    else:
      raise ValueError("Unsupported clustering method. Choose from 'kmeans', 'kmeans++', 'spectral', or 'hierarchical'.")

    prototypes[start:end, :] = centroids
    classes[start:end] = cls

  if cache is not False:
    with open(cache, "wb") as f:
      pickle.dump((prototypes, classes), f, protocol=pickle.HIGHEST_PROTOCOL)

  return prototypes, classes
