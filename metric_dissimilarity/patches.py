"""Patch generation and image-to-tensor conversion utilities.

Provides functions for extracting patches from images with configurable
overlap and minimum patch counts, converting image batches to PyTorch
tensors, and a Dataset class for patch-based data loading.
"""

import math
import random

import numpy as np
import torch
import torchvision


def gen_patches(img, patch_size, min_patches=None, regular=True):
  """
  Generates patches from an input image with an optional minimum patch count and patch distribution.

  Parameters
  ----------
  img : numpy.ndarray
      Input image as a NumPy array of shape (height, width, channels).
  patch_size : tuple
      A tuple (patch_height, patch_width) specifying the size of each patch.
  min_patches : int, optional
      The minimum number of patches required. Defaults to None.
  regular : bool, optional
      If True, generates a regular grid of patches. If False, randomly drops some patches to match `min_patches`. Defaults to True.

  Returns
  -------
  numpy.ndarray
      A 4D NumPy array of shape (number_of_patches, patch_height, patch_width, channels) containing the generated patches.
  """

  input_shape = img.shape

  # Calculate the minimum number of rows and columns of patches to cover the image
  n_rows = math.ceil(input_shape[0] / patch_size[0])
  n_cols = math.ceil(input_shape[1] / patch_size[1])
  n_patches = n_rows * n_cols

  # Adjust rows/cols to ensure at least min_patches are created
  if min_patches is not None:
    while min_patches > n_patches:
      row_ratio = input_shape[0] / n_rows / patch_size[0]
      col_ratio = input_shape[1] / n_cols / patch_size[1]
      if row_ratio > col_ratio:
        n_rows += 1
      else:
        n_cols += 1
      n_patches = n_rows * n_cols

  # Calculate overlap between patches
  row_overlap = math.ceil(((patch_size[0] * n_rows) - input_shape[0]) / (n_rows - 1))
  col_overlap = math.ceil(((patch_size[1] * n_cols) - input_shape[1]) / (n_cols - 1))

  # Generate all starting pixels, except the last one
  row_patches = np.arange(0, input_shape[0], patch_size[0] - row_overlap)[0:(n_rows - 1)]
  col_patches = np.arange(0, input_shape[1], patch_size[1] - col_overlap)[0:(n_cols - 1)]

  # Create the last starting pixel manually to avoid going larger than the input image
  row_patches = np.append(row_patches, input_shape[0] - patch_size[0])
  col_patches = np.append(col_patches, input_shape[1] - patch_size[1])

  row_patches = [(i, i + patch_size[0]) for i in row_patches]
  col_patches = [(i, i + patch_size[1]) for i in col_patches]

  patches_indices = [(i, j) for i in row_patches for j in col_patches]

  # If not regular, drop some patches to match min_patches
  if not regular:
    n_drop = n_patches - min_patches
    if n_drop > 0:
      drop_indices = random.sample(range(n_patches), n_drop)
      patches_indices = [patches_indices[i] for i in range(n_patches) if i not in drop_indices]
      n_patches = min_patches

  patches = np.zeros((n_patches, patch_size[0], patch_size[1], input_shape[2]), dtype=np.float32)

  for patch_i in range(n_patches):
    row, col = patches_indices[patch_i]
    patches[patch_i] = img[row[0]:row[1], col[0]:col[1], :]

  if img.dtype == "uint8":
    patches = (patches / 255).astype(np.float32)

  return patches


def img_to_torch(batch, device=None):
  """
  Converts a batch of images to PyTorch tensors and optionally moves them to a specified device.

  Parameters
  ----------
  batch : list of numpy.ndarray
      A list of images where each image is a NumPy array.
  device : torch.device, optional
      The device to move the output tensors to. Defaults to None.

  Returns
  -------
  torch.Tensor
      A batch of images converted to a PyTorch tensor.
  """

  transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
  ])

  batch = torch.stack([transform(im) for im in batch])

  if device is not None:
    batch = batch.to(device)

  return batch


class PatchData(torch.utils.data.Dataset):
  """
  A custom Dataset class for generating and retrieving image patches as PyTorch tensors.

  Parameters
  ----------
  data : numpy.ndarray
      The input dataset containing images from which patches will be generated.
  patch_size : tuple
      A tuple (height, width) indicating the size of the patches to be generated.
  device : torch.device, optional
      The device to move the output tensors to. Defaults to None.
  """

  def __init__(self, data, patch_size, device=None):
    self.data = data
    self.device = device
    self.patch_size = patch_size
    self.size = self.data.shape[0]

  def __getitem__(self, index):
    patches = gen_patches(self.data[index], self.patch_size)
    return img_to_torch(patches, self.device)

  def __len__(self):
    return self.size
