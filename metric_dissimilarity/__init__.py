"""Metric dissimilarity learning framework.

Provides tools for training dissimilarity-based models, generating
embeddings, selecting prototypes, and computing dissimilarity space
and vector representations for image and feature classification.
"""

from .patches import gen_patches, img_to_torch, PatchData
from .data import pair_batch, triplet_batch, MulticlassDataset
from .losses import DissimilarityNTXentLoss, TripletDissimilarityLoss
from .models import (
  VGG32,
  Network,
  ProjectionHead,
  ContrastiveModel,
  TripletModel,
)
from .training import train
from .embedding import generate_embedding, umap_projection
from .prototypes import compute_centroids, compute_prototypes
from .representations import (
  space_representation,
  vector_representation,
  vector_to_class,
  cosine_distance,
  tradt_space_representation,
  tradt_vector_representation,
)
from .osr import (
  compute_msp,
  compute_mls,
  compute_mds,
  closed_accuracy,
  open_auroc,
  openauc,
)

__all__ = [
  # patches
  "gen_patches",
  "img_to_torch",
  "PatchData",
  # data
  "pair_batch",
  "triplet_batch",
  "MulticlassDataset",
  # losses
  "DissimilarityNTXentLoss",
  "TripletDissimilarityLoss",
  # models
  "VGG32",
  "Network",
  "ProjectionHead",
  "ContrastiveModel",
  "TripletModel",
  # training
  "train",
  # embedding
  "generate_embedding",
  "umap_projection",
  # prototypes
  "compute_centroids",
  "compute_prototypes",
  # representations
  "space_representation",
  "vector_representation",
  "vector_to_class",
  "cosine_distance",
  "tradt_space_representation",
  "tradt_vector_representation",
  # osr
  "compute_msp",
  "compute_mls",
  "compute_mds",
  "closed_accuracy",
  "open_auroc",
  "openauc",
]
