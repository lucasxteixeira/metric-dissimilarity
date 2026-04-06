"""Batch generation and dataset classes for dissimilarity training.

Provides functions for generating contrastive pairs and triplet batches,
along with a standard multiclass Dataset class.
"""

import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def pair_batch(batch_size, X, Y, augments, encoded=False, size=None, device=None):
  """
  Generates a batch of pairs (images or encoded samples) and their corresponding classes.

  Parameters
  ----------
  batch_size : int
      The number of pairs to generate.
  X : numpy.ndarray
      The input data (either image dataset or already encoded samples).
  Y : numpy.ndarray
      The class labels for the input images.
  augments : callable
      A function that applies augmentations to the input images.
  encoded : bool, optional
      If True, the input `X` is already encoded and augmentation is skipped. Defaults to False.
  size : tuple, optional
      The desired image size to resize to. Defaults to None.
  device : torch.device, optional
      The device to move the output tensors to. Defaults to None.

  Returns
  -------
  list of torch.Tensor
      Two tensors for the image pairs and a tensor for the corresponding class labels.
  """

  classes = np.random.choice(np.unique(Y), size=batch_size, replace=True)

  if encoded:
    output_shape = (batch_size, X.shape[1])
  else:
    n_channels = 3 if len(X.shape) == 4 else 1
    if size is None:
      output_shape = (batch_size, n_channels, X.shape[1], X.shape[2])
    else:
      output_shape = (batch_size, n_channels, size[0], size[1])

  pairs = [torch.zeros(output_shape, dtype=torch.float32) for _ in range(2)]
  pairs.append(torch.from_numpy(classes))

  for i in range(batch_size):
    choices = np.where(Y == classes[i])[0]

    idx_A = np.random.choice(choices)
    idx_B = np.random.choice(choices)

    if not encoded:
      img_A = augments(image=X[idx_A])["image"]
      img_B = augments(image=X[idx_B])["image"]

      pairs[0][i] = img_A / 255.
      pairs[1][i] = img_B / 255.
    else:
      pairs[0][i] = torch.tensor(X[idx_A])
      pairs[1][i] = torch.tensor(X[idx_B])

  if device is not None:
    pairs = [t.to(device) for t in pairs]

  return pairs


def triplet_batch(batch_size, X, Y, augments, encoded=False, size=None, train_embeddings=None, hardness=50, anchors=None, device=None):
  """
  Generates a batch of triplets (images or encoded samples) for training.

  Parameters
  ----------
  batch_size : int
      The number of triplets to generate.
  X : numpy.ndarray
      The input dataset containing images.
  Y : numpy.ndarray
      The class labels corresponding to each image.
  augments : callable
      A function that applies augmentations to the input images.
  encoded : bool, optional
      If True, the input `X` is already encoded and augmentation is skipped. Defaults to False.
  size : tuple, optional
      The desired image size to resize to. Defaults to None.
  train_embeddings : numpy.ndarray, optional
      Embeddings for hard negative mining. Defaults to None.
  hardness : int, optional
      The percentile for hard positive/negative selection. Defaults to 50.
  anchors : numpy.ndarray, optional
      Boolean mask for valid anchor indices. If None, random anchors are selected. Defaults to None.
  device : torch.device, optional
      The device to move the output tensors to. Defaults to None.

  Returns
  -------
  list of torch.Tensor
      A list containing the generated triplets (anchor, positive, negative).
  """

  classes = np.random.choice(np.unique(Y), size=batch_size, replace=True)

  if encoded:
    output_shape = (batch_size, X.shape[1])
  else:
    n_channels = 3 if len(X.shape) == 4 else 1
    if size is None:
      output_shape = (batch_size, n_channels, X.shape[1], X.shape[2])
    else:
      output_shape = (batch_size, n_channels, size[0], size[1])

  triplets = [torch.zeros(output_shape, dtype=torch.float32) for _ in range(3)]

  for i in range(batch_size):

    # Anchor mining
    if anchors is not None:
      anchor_index = np.random.choice(np.where(anchors)[0])
      anchor_class = Y[anchor_index]
      anchor_choices = np.where(Y == anchor_class)[0]
    else:
      anchor_class = classes[i]
      anchor_choices = np.where(Y == anchor_class)[0]
      anchor_index = np.random.choice(anchor_choices)

    # Offline triplet mining with embeddings
    if train_embeddings is not None:
      positive_dist = np.linalg.norm(train_embeddings[anchor_index,:] - train_embeddings[anchor_choices,:], axis=1)
      valid_positive = positive_dist >= np.percentile(positive_dist, hardness)
      positive_index = np.random.choice(anchor_choices[valid_positive])

      negative_choices = np.where(Y != anchor_class)[0]
      negative_dist = np.linalg.norm(train_embeddings[anchor_index,:] - train_embeddings[negative_choices,:], axis=1)
      valid_negative = negative_dist <= np.percentile(negative_dist, 100 - hardness)
      negative_index = np.random.choice(negative_choices[valid_negative])

    # Random triplets
    else:
      positive_index = np.random.choice(anchor_choices)

      negative_choices = np.where(Y != anchor_class)[0]
      negative_index = np.random.choice(negative_choices)

    if not encoded:
      anchor = augments(image=X[anchor_index])["image"]
      positive = augments(image=X[positive_index])["image"]
      negative = augments(image=X[negative_index])["image"]

      triplets[0][i] = anchor / 255.
      triplets[1][i] = positive / 255.
      triplets[2][i] = negative / 255.
    else:
      triplets[0][i] = torch.tensor(X[anchor_index])
      triplets[1][i] = torch.tensor(X[positive_index])
      triplets[2][i] = torch.tensor(X[negative_index])

  if device is not None:
    triplets = [t.to(device) for t in triplets]

  return triplets


class MulticlassDataset(torch.utils.data.Dataset):
  """
  Custom Dataset class for loading images and their corresponding labels.

  Parameters
  ----------
  X : numpy.ndarray
      The input dataset containing images.
  Y : numpy.ndarray
      The class labels corresponding to each image.
  augments : callable
      A function that applies augmentations to the input images.
  encoded : bool, optional
      If True, the input `X` is already encoded and augmentation is skipped. Defaults to False.
  """

  def __init__(self, X, Y, augments, encoded=False):
    self.X = X
    self.Y = Y
    self.augments = augments
    self.encoded = encoded

  def __len__(self):
     return len(self.X)

  def __getitem__(self, idx):
    image = self.X[idx]
    label = self.Y[idx]

    if not self.encoded:
      image = self.augments(image = image)["image"]
    else:
      image = ToTensorV2()(image=image)["image"]

    image = image / 255.

    return image, label
