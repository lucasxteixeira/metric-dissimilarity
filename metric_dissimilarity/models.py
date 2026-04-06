"""Neural network architectures for dissimilarity learning.

Provides backbone networks, embedding heads, projection heads,
and full contrastive/triplet dissimilarity models.
"""

import torch
import torchvision


class VGG32(torch.nn.Module):
  """A custom VGG-style network for 32x32 images."""

  def __init__(self):
    super(VGG32, self).__init__()

    self.features = torch.nn.Sequential(
      torch.nn.Dropout(0.2),

      torch.nn.Conv2d(3, 32, 3, padding = 1),
      torch.nn.BatchNorm2d(32),
      torch.nn.LeakyReLU(0.2),

      torch.nn.Conv2d(32, 64, 3, padding = 1),
      torch.nn.BatchNorm2d(64),
      torch.nn.LeakyReLU(0.2),

      torch.nn.Conv2d(64, 128, 3, stride = 2, padding = 1),
      torch.nn.BatchNorm2d(128),
      torch.nn.LeakyReLU(0.2),

      torch.nn.Dropout(0.2),

      torch.nn.Conv2d(128, 256, 3, padding = 1),
      torch.nn.BatchNorm2d(256),
      torch.nn.LeakyReLU(0.2),

      torch.nn.Conv2d(256, 512, 3, padding = 1),
      torch.nn.BatchNorm2d(512),
      torch.nn.LeakyReLU(0.2),

      torch.nn.Conv2d(512, 1024, 3, stride = 2, padding = 1),
      torch.nn.BatchNorm2d(1024),
      torch.nn.LeakyReLU(0.2),

      torch.nn.Dropout(0.2),

      torch.nn.Conv2d(1024, 1024, 3, padding = 1),
      torch.nn.BatchNorm2d(1024),
      torch.nn.LeakyReLU(0.2),

      torch.nn.Conv2d(1024, 1024, 3, padding = 1),
      torch.nn.BatchNorm2d(1024),
      torch.nn.LeakyReLU(0.2),

      torch.nn.Conv2d(1024, 1024, 3, stride = 2, padding = 1),
      torch.nn.BatchNorm2d(1024),
      torch.nn.LeakyReLU(0.2),

      torch.nn.AdaptiveAvgPool2d((1, 1)),

      torch.nn.Flatten(),
      torch.nn.Linear(1024, 128)
    )

    self.fc = torch.nn.Linear(128, 1)

    self.feature_dim = 128

  def forward(self, x):
    x = self.features(x)
    x = self.fc(x)
    return x


def _get_backbone(backbone, pretrained=True):
  """
  Get the backbone model and its feature size.

  Parameters
  ----------
  backbone : str
      The type of backbone model to use.
      Tested with 'resnet50', 'resnet101', 'resnet152',
        'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
        'convnext_small', 'convnext_base', 'convnext_large',
        and 'mobilenet_v2'.
  pretrained : bool or str, optional
      If False, no pre-trained weights are loaded.
      If True, uses default pre-trained weights.
      If a string, uses weights from the specified path.
      Defaults to True.

  Returns
  -------
  network : torch.nn.Module
      The backbone model.
  extra_layers : list
      A list of extra layers to be added to the model.
  feature_size : int
      The size of the feature vector output by the backbone model.
  """

  if backbone == "VGG32":
    network = VGG32()
    extra_layers = []
    feature_size = network.feature_dim
    return network, [], feature_size

  model_fn = getattr(torchvision.models, backbone, None)

  # Load pretrained weights
  if isinstance(pretrained, str):
    raw_weights = torch.load(pretrained, weights_only=True)

    weights = {}
    for k, v in raw_weights.items():
      if k.startswith("network.classifier"):
        continue
      new_key = k
      if k.startswith("network."):
        new_key = k[len("network."):]
      weights[new_key] = v

    network = model_fn(weights=None)
    network.load_state_dict(weights, strict=False)
    print("Loaded pretrained weights from", pretrained)

  elif pretrained:
    network = model_fn(weights="DEFAULT")
    print("Loaded default pretrained weights from torchvision")
  else:
    network = model_fn(weights=None)

  extra_layers = []

  if hasattr(network, "fc"):
    feature_size = network.fc.in_features

  elif hasattr(network, "classifier"):
    cls = network.classifier
    if backbone.startswith("efficientnet"):
      feature_size = cls[1].in_features
    elif backbone.startswith("mobilenet"):
      feature_size = cls[0].in_features
    elif backbone.startswith("convnext"):
      extra_layers.extend([cls[0], cls[1]])
      feature_size = cls[2].in_features
    else:
      raise RuntimeError(f"Unexpected classifier head for backbone {backbone!r}")
  else:
    raise RuntimeError(f"Could not find classifier head for backbone {backbone!r}")

  return network, extra_layers, feature_size


class Network(torch.nn.Module):
  """
  Base network for feature extraction with an embedding head.

  Parameters
  ----------
  embeddingsize : int
      The size of the output embedding vector.
  backbone : str
      Backbone model name.
  pretrained : bool or str, optional
      Pre-trained weights configuration. Defaults to True.
  hidden_layers : list of int, optional
      Hidden layer sizes for the embedding head.
      Defaults to [embeddingsize*4, embeddingsize*2].
  num_classes : int, optional
      Number of classes for the classification head (used in cross-entropy warmup). Defaults to None.
  """

  def __init__(self, embeddingsize, backbone, pretrained=True, hidden_layers=None, num_classes=None):
    super(Network, self).__init__()

    self.network, pre_head_layers, feature_size = _get_backbone(backbone, pretrained=pretrained)

    # Freeze the backbone network
    for param in self.network.parameters():
      param.requires_grad = False

    if hidden_layers is None:
      hidden_layers = [embeddingsize*4, embeddingsize*2]

    # Create the embedding head
    layers = []
    layers.extend(pre_head_layers)

    if hidden_layers:
      for hidden_size in hidden_layers:
        layers.append(torch.nn.Linear(feature_size, hidden_size))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Dropout(0.2))
        feature_size = hidden_size

    layers.append(torch.nn.Linear(feature_size, embeddingsize))

    # Replace the original classifier layers with the embedding head
    if hasattr(self.network, "fc"):
      self.network.fc = torch.nn.Sequential(*layers)
    else:
      self.network.classifier = torch.nn.Sequential(*layers)

    # Optional classifier head for cross-entropy warmup
    if num_classes is not None:
      self.clf = torch.nn.Sequential(
        torch.nn.Linear(embeddingsize, num_classes)
      )

  def forward(self, x, mode="embedding"):
    x = self.network(x)

    if mode == "embedding":
      return torch.nn.functional.normalize(x, p=2, dim=1)

    elif mode == "classifier":
      if self.clf is None:
        raise RuntimeError("Classifier head not defined. Set num_classes to add a classifier head.")
      return self.clf(x)

    else:
      raise ValueError("Invalid mode. Choose either 'embedding' or 'classifier'.")


class ProjectionHead(torch.nn.Module):
  """
  Projection head for computing dissimilarity values from embedding pairs.

  Parameters
  ----------
  embeddingsize : int
      The size of the input embedding vector.
  hidden_layers : list of int, optional
      Hidden layer sizes. Defaults to [embeddingsize//2, embeddingsize//4].
  output_size : int, optional
      The size of the output layer. Defaults to 1.
  """

  def __init__(self, embeddingsize, hidden_layers=None, output_size=1):
    super(ProjectionHead, self).__init__()

    if hidden_layers is None:
      hidden_layers = [embeddingsize//2, embeddingsize//4]

    layers = []
    input_size = embeddingsize

    for hidden_size in hidden_layers:
      layers.append(torch.nn.Linear(input_size, hidden_size))
      layers.append(torch.nn.ReLU(inplace=True))
      input_size = hidden_size

    layers.append(torch.nn.Linear(input_size, output_size))
    self.projection_head = torch.nn.Sequential(*layers)

  def forward(self, x1, x2):
    return self.projection_head(torch.abs(x1 - x2))


class ContrastiveModel(torch.nn.Module):
  """
  Contrastive dissimilarity model combining a base network and projection head.

  Parameters
  ----------
  embeddingsize : int
      The size of the output embedding vector.
  backbone : str
      Backbone model name.
  projection_head : list of int, optional
      Projection head hidden layers. Defaults to None.
  top_layers : list of int, optional
      Base network top layers. Defaults to None.
  encoded : bool, optional
      If True, the input is already encoded (skip embedding). Defaults to False.
  pretrained : bool or str, optional
      Pre-trained weights configuration. Defaults to True.
  num_classes : int, optional
      Number of classes for the classification head. Defaults to None.
  """

  def __init__(self, embeddingsize, backbone, projection_head=None, top_layers=None, encoded=False, pretrained=True, num_classes=None):
    super(ContrastiveModel, self).__init__()

    self.network = None
    if not encoded:
      self.network = Network(embeddingsize, backbone=backbone, hidden_layers=top_layers, pretrained=pretrained, num_classes=num_classes)

    self.projection_head = ProjectionHead(embeddingsize, hidden_layers=projection_head)

  def forward(self, x1, x2):

    if self.network is not None:
      x1 = self.network(x1)
      x2 = self.network(x2)

    if self.training:
      batch_size = x1.shape[0]

      x = torch.cat([x1, x2])
      x1 = torch.tile(x, [batch_size * 2, 1])
      x2 = torch.repeat_interleave(x, batch_size * 2, dim=0)

      dissimilarity = self.projection_head(x1, x2)
      dissimilarity = torch.reshape(dissimilarity, (batch_size * 2, -1))
    else:
      dissimilarity = self.projection_head(x1, x2)

    return dissimilarity

  def freeze_network(self):
    if self.network is not None:
      for param in self.network.parameters():
        param.requires_grad = False

  def unfreeze_network(self):
    if self.network is not None:
      for param in self.network.parameters():
        param.requires_grad = True


class TripletModel(torch.nn.Module):
  """
  Triplet dissimilarity model combining a base network and projection head.

  Parameters
  ----------
  embeddingsize : int
      The size of the output embedding vector.
  backbone : str
      Backbone model name.
  projection_head : list of int, optional
      Projection head hidden layers. Defaults to None.
  top_layers : list of int, optional
      Base network top layers. Defaults to None.
  encoded : bool, optional
      If True, the input is already encoded (skip embedding). Defaults to False.
  pretrained : bool or str, optional
      Pre-trained weights configuration. Defaults to True.
  num_classes : int, optional
      Number of classes for the classification head. Defaults to None.
  """

  def __init__(self, embeddingsize, backbone, projection_head=None, top_layers=None, encoded=False, pretrained=True, num_classes=None):
    super(TripletModel, self).__init__()

    self.network = None
    if not encoded:
      self.network = Network(embeddingsize, backbone=backbone, hidden_layers=top_layers, pretrained=pretrained, num_classes=num_classes)

    self.projection_head = ProjectionHead(embeddingsize, hidden_layers=projection_head)

  def forward(self, anchor, positive, negative):

    if self.network is not None:
      anchor = self.network(anchor)
      positive = self.network(positive)
      negative = self.network(negative)

    pos_dissimilarity = self.projection_head(anchor, positive)
    neg_dissimilarity = self.projection_head(anchor, negative)

    return pos_dissimilarity, neg_dissimilarity

  def freeze_network(self):
    if self.network is not None:
      for param in self.network.parameters():
        param.requires_grad = False

  def unfreeze_network(self):
    if self.network is not None:
      for param in self.network.parameters():
        param.requires_grad = True
