"""Dissimilarity space and vector representations.

Provides functions for computing metric-learned and traditional
dissimilarity representations from embeddings and prototypes.
"""

import os
import pickle

import numpy as np
import torch

from .patches import gen_patches, img_to_torch


def space_representation(model, encoded, X_prot, cache="space.pkl"):
  """
  Computes the metric dissimilarity space for a given dataset.

  Parameters
  ----------
  model : torch.nn.Module
      The trained model with a projection head for computing dissimilarity.
  encoded : np.ndarray
      Encoded data of shape (num_samples, num_features).
  X_prot : np.ndarray
      Prototypes of shape (num_prototypes, num_features).
  cache : str or False, optional
      File path for caching. Set to False to disable. Defaults to 'space.pkl'.

  Returns
  -------
  np.ndarray
      The dissimilarity space representation.
  """

  if cache is not False and os.path.isfile(cache):
    with open(cache, "rb") as f:
      space = pickle.load(f)
    return space

  device = next(model.parameters()).device

  X_prot = torch.from_numpy(X_prot).to(device)
  n_prot = X_prot.shape[0]

  space = []

  for idx in range(encoded.shape[0]):
    local_x = encoded[[idx], :].astype(np.float32)
    local_x = np.repeat(local_x, n_prot, axis=0)
    local_x = torch.from_numpy(local_x).to(device)

    diss = model.projection_head(local_x, X_prot).squeeze().cpu().detach().numpy()
    space.append(diss)

  space = np.array(space)

  if cache is not False:
    with open(cache, "wb") as f:
      pickle.dump(space, f, protocol=pickle.HIGHEST_PROTOCOL)

  return space


def vector_representation(model, X, Y, X_prot, Y_prot, patch_size=None, variations=20, cache="vector.pkl"):
  """
  Computes the metric dissimilarity vector for given datasets.

  Parameters
  ----------
  model : torch.nn.Module
      The trained model with a projection head for computing dissimilarity.
  X : np.ndarray
      The input data.
  Y : np.ndarray
      The class labels for the input data.
  X_prot : np.ndarray
      The prototypes data.
  Y_prot : np.ndarray
      The class labels for the prototypes.
  patch_size : tuple, optional
      The size of patches to generate from images. Defaults to None.
  variations : int, optional
      Minimum number of variations per input. Defaults to 20.
  cache : str or False, optional
      File path for caching. Set to False to disable. Defaults to 'vector.pkl'.

  Returns
  -------
  np.ndarray
      Dissimilarity vector representation.
  np.ndarray
      Corresponding labels.
  """

  if cache is not False and os.path.isfile(cache):
    with open(cache, "rb") as f:
      X_vector, Y_vector = pickle.load(f)
    return X_vector, Y_vector

  device = next(model.parameters()).device

  Y_vector = np.equal.outer(Y, Y_prot).ravel()

  X_vector = []

  # When the network is None, the input is already encoded.
  # Enable training mode to activate dropout for generating variations.
  if model.network is None:
    model.train()

  for idx in range(X.shape[0]):

    if model.network is None:
      number_prototypes = X_prot.shape[0]

      local_encodings = torch.from_numpy(np.float32(X[idx])).to(device)
      local_encodings = torch.tile(local_encodings, [number_prototypes * variations, 1])

      local_prototypes = np.repeat(X_prot, variations, axis=0)
      local_prototypes = torch.from_numpy(local_prototypes).to(device)

    else:
      local_patches = img_to_torch(gen_patches(X[idx], patch_size, min_patches=variations * 5, regular=False), device=device)
      patch_encodings = model.network(local_patches)

      # Average groups of patches for more stable encodings
      patch_encodings = torch.mean(torch.stack(patch_encodings.split(5)), dim=1, dtype=torch.float32)

      number_patches = patch_encodings.shape[0]
      number_prototypes = X_prot.shape[0]

      local_encodings = torch.tile(patch_encodings, [number_prototypes, 1])
      local_prototypes = np.repeat(X_prot, number_patches, axis=0)
      local_prototypes = torch.from_numpy(local_prototypes).to(device)

    diss_vec = model.projection_head(local_encodings, local_prototypes)
    diss_vec = np.array(np.split(diss_vec.cpu().detach().numpy(), number_prototypes))
    X_vector.append(diss_vec)

  X_vector = np.reshape(X_vector, (len(Y) * number_prototypes, -1))

  if cache is not False:
    with open(cache, "wb") as f:
      pickle.dump((X_vector, Y_vector), f, protocol=pickle.HIGHEST_PROTOCOL)

  return X_vector, Y_vector


def vector_to_class(probs, y_true, y_prot):
  """
  Transforms metric dissimilarity vector representation back into multiclass predictions.

  Parameters
  ----------
  probs : np.ndarray
      Predicted probabilities of shape (n_samples * n_prototypes, 2).
  y_true : np.ndarray
      True class labels for the test data.
  y_prot : np.ndarray
      Prototype labels.

  Returns
  -------
  np.ndarray
      Predicted class labels.
  np.ndarray
      Aggregated prediction probabilities per class.
  """

  probs = np.reshape(probs[:, 1], (y_true.shape[0], -1))

  prot_per_class = np.bincount(y_prot).max()

  probs = np.reshape(probs, (y_true.shape[0], -1, prot_per_class))
  probs = np.max(probs, axis=-1)

  preds = np.argmax(probs, axis=1)

  return preds, probs


# --- Traditional dissimilarity representations ---


def cosine_distance(x, y):
  """
  Computes the cosine similarity between two sets of vectors.

  Parameters
  ----------
  x : np.ndarray
      The first set of vectors.
  y : np.ndarray
      The second set of vectors.

  Returns
  -------
  np.ndarray
      The cosine similarity matrix.
  """

  norm_x = x / np.linalg.norm(x, axis=1, keepdims=True)
  norm_y = y / np.linalg.norm(y, axis=1, keepdims=True)

  return np.matmul(norm_x, norm_y.T)


def tradt_space_representation(encoded, X_prot, distance="euclidean", cache="tradt-space.pkl"):
  """
  Computes the traditional dissimilarity space.

  Parameters
  ----------
  encoded : np.ndarray
      Encoded data of shape (num_samples, num_features).
  X_prot : np.ndarray
      Prototypes of shape (num_prototypes, num_features).
  distance : str, optional
      Distance metric: 'euclidean' or 'cosine'. Defaults to 'euclidean'.
  cache : str or False, optional
      File path for caching. Set to False to disable. Defaults to 'tradt-space.pkl'.

  Returns
  -------
  np.ndarray
      The traditional dissimilarity space representation.
  """

  if cache is not False and os.path.isfile(cache):
    with open(cache, "rb") as f:
      space = pickle.load(f)
    return space

  if distance == "euclidean":
    encoded = encoded[:,np.newaxis,:]
    space = np.linalg.norm(encoded - X_prot, axis=2)

  elif distance == "cosine":
    space = cosine_distance(encoded, X_prot)

  else:
    raise ValueError("Unsupported distance metric. Choose either 'euclidean' or 'cosine'.")

  if cache is not False:
    with open(cache, "wb") as f:
      pickle.dump(space, f, protocol=pickle.HIGHEST_PROTOCOL)

  return space


def tradt_vector_representation(encoded, Y, X_prot, Y_prot, cache="tradt-vector.pkl"):
  """
  Computes the traditional dissimilarity vector.

  Parameters
  ----------
  encoded : np.ndarray
      Encoded data of shape (num_samples, num_features).
  Y : np.ndarray
      Class labels for the input data.
  X_prot : np.ndarray
      Prototypes data.
  Y_prot : np.ndarray
      Class labels for the prototypes.
  cache : str or False, optional
      File path for caching. Set to False to disable. Defaults to 'tradt-vector.pkl'.

  Returns
  -------
  np.ndarray
      The traditional dissimilarity vector representation.
  np.ndarray
      Corresponding labels.
  """

  if cache is not False and os.path.isfile(cache):
    with open(cache, "rb") as f:
      X_vector, Y_vector = pickle.load(f)
    return X_vector, Y_vector

  Y_vector = np.equal.outer(Y, Y_prot).ravel()

  embedding = encoded.shape[1]

  encoded = encoded[:,np.newaxis,:]
  X_prot = X_prot[np.newaxis, :, :]

  X_vector = np.abs(encoded - X_prot).reshape(-1, embedding)

  if cache is not False:
    with open(cache, "wb") as f:
      pickle.dump((X_vector, Y_vector), f, protocol=pickle.HIGHEST_PROTOCOL)

  return X_vector, Y_vector
