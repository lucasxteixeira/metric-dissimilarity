"""Main training loop for dissimilarity models.

Provides the unified training function supporting contrastive and
triplet models with optional cross-entropy warmup, projection head
warmup, and triplet mining phases.
"""

import os

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder

from .patches import PatchData
from .data import pair_batch, triplet_batch, MulticlassDataset
from .losses import DissimilarityNTXentLoss, TripletDissimilarityLoss
from .models import ContrastiveModel, TripletModel


def train(X, Y, model_type, model_file, backbone,

          # Common model parameters
          embeddingsize=128, patch_size=None, projection_head=None, top_layers=None, pretrained=True, augments=None,
          batch=32, iterations=10000, lr=0.001,

          # Common warmup parameters
          batch_warmup=64,

          # Cross-entropy warmup parameters
          clf_warmup=False, clf_warmup_epochs=10, clf_epochs=50, clf_warmup_lr=0.01, clf_lr=0.001,

          # Projection head warmup parameters
          warmup_iterations=5000, lr_warmup=0.01,

          # Contrastive model parameters
          temperature_warmup=0.5, temperature=0.5,

          # Triplet model parameters
          alpha_warmup=1.0, alpha=1.0,
          triplet_mining=False, mining_iterations=10000, mining_hardness=50, mining_lr=0.001):
  """
  Train a contrastive or triplet dissimilarity model.

  Parameters
  ----------
  X : numpy.ndarray
      The input dataset containing images.
  Y : numpy.ndarray
      The class labels corresponding to each image.
  model_type : str
      The type of model to train: 'contrastive' or 'triplet'.
  model_file : str
      The file path to save the trained model.
  backbone : str
      The backbone model to use for feature extraction.
  embeddingsize : int, optional
      The size of the output embedding vector. Defaults to 128.
  patch_size : tuple, optional
      The size of patches to generate from images. Defaults to None.
  projection_head : list of int, optional
      Hidden layer sizes for the projection head. Defaults to None.
  top_layers : list of int, optional
      Hidden layer sizes for the base network. Defaults to None.
  pretrained : bool or str, optional
      Pre-trained weights configuration. Defaults to True.
  augments : callable, optional
      Augmentation pipeline. Defaults to None (uses built-in defaults).
  batch : int, optional
      Batch size for training. Defaults to 32.
  iterations : int, optional
      Number of training iterations. Defaults to 10000.
  lr : float, optional
      Learning rate. Defaults to 0.001.
  batch_warmup : int, optional
      Batch size for the warmup phase. Defaults to 64.
  clf_warmup : bool, optional
      If True, performs a cross-entropy warmup phase. Defaults to False.
  clf_warmup_epochs : int, optional
      Epochs for the cross-entropy warmup phase. Defaults to 10.
  clf_epochs : int, optional
      Epochs for the main cross-entropy training phase. Defaults to 50.
  clf_warmup_lr : float, optional
      Learning rate for cross-entropy warmup. Defaults to 0.01.
  clf_lr : float, optional
      Learning rate for main cross-entropy training. Defaults to 0.001.
  warmup_iterations : int, optional
      Iterations for the projection head warmup. Defaults to 5000.
  lr_warmup : float, optional
      Learning rate for projection head warmup. Defaults to 0.01.
  temperature_warmup : float, optional
      Temperature for contrastive warmup. Defaults to 0.5.
  temperature : float, optional
      Temperature for contrastive training. Defaults to 0.5.
  alpha_warmup : float, optional
      Margin for triplet warmup. Defaults to 1.0.
  alpha : float, optional
      Margin for triplet training. Defaults to 1.0.
  triplet_mining : bool, optional
      If True, performs triplet mining after training. Defaults to False.
  mining_iterations : int, optional
      Iterations for triplet mining. Defaults to 10000.
  mining_hardness : int, optional
      Hardness percentile for triplet mining. Defaults to 50.
  mining_lr : float, optional
      Learning rate for triplet mining. Defaults to 0.001.

  Returns
  -------
  torch.nn.Module
      The trained model.
  """

  print(f"Model file: {model_file}")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = None

  # When embeddingsize is None, the input is already encoded
  encoded = False
  if embeddingsize is None:
    encoded = True
    embeddingsize = X.shape[1]

  elif augments is None:
    if patch_size is None:
      patch_size = (X.shape[2], X.shape[3])

    augments = A.Compose([
      A.RandomCrop(patch_size[0], patch_size[1]),
      A.VerticalFlip(),
      A.HorizontalFlip(),
      A.Rotate(),
      A.GaussianBlur(),
      A.RandomBrightnessContrast(),
      ToTensorV2()
    ])

  # Encode labels for cross-entropy compatibility
  Y = LabelEncoder().fit_transform(Y)

  num_classes = len(np.unique(Y))

  if model_type == "contrastive":
    model_fn = ContrastiveModel
    loss_warmup_fn = DissimilarityNTXentLoss(temperature_warmup)
    loss_fn = DissimilarityNTXentLoss(temperature)
  elif model_type == "triplet":
    model_fn = TripletModel
    loss_warmup_fn = TripletDissimilarityLoss(alpha_warmup)
    loss_fn = TripletDissimilarityLoss(alpha)
  else:
    raise ValueError("Invalid model type. Choose either 'contrastive' or 'triplet'.")

  # Load pre-trained model if available
  if os.path.isfile(model_file):
    print("Loading pre-trained model...")
    model = model_fn(embeddingsize, backbone,
                     projection_head=projection_head, top_layers=top_layers, encoded=encoded,
                     pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_file, weights_only=True), strict=True)
    model.to(device)

  # Train a new model
  if model is None:
    print("Training a new model...")

    model = model_fn(embeddingsize, backbone,
                     projection_head=projection_head, top_layers=top_layers, encoded=encoded,
                     pretrained=pretrained, num_classes=num_classes)

    model.to(device)
    model.train()

    # --- Cross-entropy warmup ---

    if clf_warmup:
      print("Cross-entropy Warmup Phase")

      backbone_net = model.network

      dataset = MulticlassDataset(X, Y, augments=augments, encoded=encoded)

      criterion = torch.nn.CrossEntropyLoss()

      if pretrained:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_warmup, shuffle=True)
        optimizer = torch.optim.SGD(backbone_net.parameters(), lr=clf_warmup_lr, momentum=0.9)

        print("Warmup top-layers")
        epoch_loss = 0
        for epoch in range(clf_warmup_epochs):
          for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = backbone_net(images, mode="classifier")

            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

          print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}")
          epoch_loss = 0

      print("Warmup backbone")

      model.unfreeze_network()

      train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
      optimizer = torch.optim.SGD(model.parameters(), lr=clf_lr, momentum=0.9)

      epoch_loss = 0
      for epoch in range(clf_epochs):
        for images, labels in train_loader:
          images = images.to(device)
          labels = labels.to(device)

          optimizer.zero_grad()

          outputs = backbone_net(images, mode="classifier")

          loss = criterion(outputs, labels)
          epoch_loss += loss.item()

          loss.backward()
          optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}")
        epoch_loss = 0

      model.freeze_network()
      pretrained = True

    # --- Projection head / top-layers warmup ---

    if pretrained:
      print("Projection Head Warmup Phase")

      optimizer = torch.optim.SGD(model.parameters(), lr=lr_warmup, momentum=0.9)

      train_loss = 0
      for epoch in range(warmup_iterations // 100):
        for _ in range(100):

          optimizer.zero_grad()

          if model_type == "contrastive":
            x1, x2, y = pair_batch(batch_warmup, X, Y, augments=augments, encoded=encoded, size=patch_size, device=device)
            outputs = model(x1, x2)
            loss = loss_warmup_fn(outputs, y)
          elif model_type == "triplet":
            anc, pos, neg = triplet_batch(batch_warmup, X, Y, augments=augments, encoded=encoded, size=patch_size, device=device)
            pos_score, neg_score = model(anc, pos, neg)
            loss = loss_warmup_fn(pos_score, neg_score)
          else:
            raise ValueError("Invalid model type. Choose either 'contrastive' or 'triplet'.")

          train_loss += loss.item()

          loss.backward()
          optimizer.step()

        print(f"Epoch {epoch + 1}, Warmup Loss: {train_loss / 100:.4f}")
        train_loss = 0

    # --- Full training phase ---

    print("Training Phase")

    model.unfreeze_network()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_loss = 0
    for epoch in range(iterations // 100):
      for _ in range(100):

        optimizer.zero_grad()

        if model_type == "contrastive":
          x1, x2, y = pair_batch(batch, X, Y, augments=augments, encoded=encoded, size=patch_size, device=device)
          outputs = model(x1, x2)
          loss = loss_fn(outputs, y)
        elif model_type == "triplet":
          anc, pos, neg = triplet_batch(batch, X, Y, augments=augments, encoded=encoded, size=patch_size, device=device)
          pos_score, neg_score = model(anc, pos, neg)
          loss = loss_fn(pos_score, neg_score)
        else:
          raise ValueError("Invalid model type. Choose either 'contrastive' or 'triplet'.")

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

      print(f"Epoch {epoch + 1}, Training Loss: {train_loss / 100:.4f}")
      train_loss = 0

    # Save the trained model
    print("Saving the trained model...")
    save_dir = os.path.dirname(model_file)
    if save_dir:
      os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), model_file)

  # --- Triplet mining phase ---

  if model_type == "triplet" and triplet_mining:
    print("Triplet Mining Phase")

    print("Extracting patches from training data...")
    model.eval()
    with torch.no_grad():
      patch_data = PatchData(X, patch_size=patch_size, device=device)
      patch_dataloader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=None, shuffle=False)
      train_embeddings = torch.stack([model.network(data) for _, data in enumerate(patch_dataloader)])

    train_embeddings = np.array(train_embeddings.cpu(), dtype=np.float32)
    train_embeddings = np.mean(train_embeddings, axis=1)

    # Find useful anchors (observations close to negative examples)
    print("Finding useful anchors for triplet mining...")
    n_obs = train_embeddings.shape[0]
    anchors = np.zeros(n_obs, dtype=bool)

    for anchor_index in range(n_obs):
      anchor_class = Y[anchor_index]
      negative_choices = np.where(Y != anchor_class)[0]
      negative_dist = np.linalg.norm(train_embeddings[anchor_index,:] - train_embeddings[negative_choices,:], axis=1)
      valid_negative = negative_dist <= alpha
      anchors[anchor_index] = np.any(valid_negative)

    if not np.any(anchors):
      print("No anchors found, try increasing alpha")
      return None

    loss_fn = TripletDissimilarityLoss(alpha)
    optimizer = torch.optim.SGD(model.parameters(), lr=mining_lr, momentum=0.9)

    model.train()
    train_loss = 0
    for epoch in range(mining_iterations // 100):
      for _ in range(100):

        optimizer.zero_grad()

        anc, pos, neg = triplet_batch(batch, X, Y, augments=augments, encoded=encoded, size=patch_size,
                                      train_embeddings=train_embeddings, hardness=mining_hardness, anchors=anchors, device=device)
        pos_score, neg_score = model(anc, pos, neg)
        loss = loss_fn(pos_score, neg_score)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

      print(f"Epoch {epoch + 1}, Training Loss: {train_loss / 100:.4f}")
      train_loss = 0

    # Save the trained model
    print("Saving the trained model...")
    save_dir = os.path.dirname(model_file)
    if save_dir:
      os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), model_file)

  model.freeze_network()
  model.eval()

  print("Model is ready for evaluation.")

  return model
