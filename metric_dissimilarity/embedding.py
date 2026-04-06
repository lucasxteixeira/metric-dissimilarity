"""Embedding generation and UMAP visualization.

Provides functions for extracting embeddings from trained models
and visualizing them with UMAP dimensionality reduction.
"""

import os
import pickle

import numpy as np
import torch
import umap
from matplotlib import pyplot as plt
import seaborn as sns

from .patches import PatchData


def generate_embedding(model, data, patch_size, cache="embedding.pkl"):
  """
  Generate embeddings for a given dataset and optionally cache them.

  Parameters
  ----------
  model : torch.nn.Module
      The trained model to be used for generating embeddings.
  data : numpy.ndarray
      The data as a NumPy array.
  patch_size : tuple of int
      The size of the patches to be generated from the images.
  cache : str or False, optional
      File path for caching embeddings. Set to False to disable caching. Defaults to 'embedding.pkl'.

  Returns
  -------
  np.ndarray
      The computed embeddings for the specified dataset.
  """

  if cache is not False and os.path.isfile(cache):
    with open(cache, "rb") as f:
      embedding = pickle.load(f)
    return embedding

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  patch_data = PatchData(data, patch_size=patch_size, device=device)
  patch_dataloader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=None, shuffle=False)

  embedding = torch.stack([
    model.network(batch_data) for _, batch_data in enumerate(patch_dataloader)
  ])

  embedding = np.array(embedding.cpu(), dtype=np.float32)
  embedding = np.mean(embedding, axis=1)

  if cache is not False:
    with open(cache, "wb") as f:
      pickle.dump(embedding, f, protocol=pickle.HIGHEST_PROTOCOL)

  return embedding


def umap_projection(encoded_X, Y, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
  """
  Visualizes high-dimensional encodings using UMAP.

  Parameters
  ----------
  encoded_X : np.ndarray
      The high-dimensional encodings of shape (num_samples, num_features).
  Y : np.ndarray or list
      The labels corresponding to each sample.
  n_neighbors : int, optional
      Number of neighbors for UMAP. Defaults to 15.
  min_dist : float, optional
      Minimum distance between points in the UMAP output. Defaults to 0.1.
  n_components : int, optional
      Number of dimensions for UMAP. Defaults to 2.
  random_state : int, optional
      Random seed for reproducibility. Defaults to 42.
  """

  assert len(encoded_X) == len(Y), "The number of samples in encoded_X and Y must be the same."

  reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state, n_jobs=1)
  trn = reducer.fit_transform(encoded_X)

  plt.figure(figsize=(10, 6))
  sns.scatterplot(
    x=trn[:, 0],
    y=trn[:, 1],
    hue=Y,
    palette=sns.color_palette("bright", len(np.unique(Y))),
    legend=False
  )
  plt.title("UMAP Projection")
  plt.xlabel("UMAP-1")
  plt.ylabel("UMAP-2")
  plt.show()
