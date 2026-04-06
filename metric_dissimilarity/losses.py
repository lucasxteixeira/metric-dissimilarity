"""Loss functions for dissimilarity learning.

Provides contrastive (NT-Xent) and triplet margin loss functions
for training dissimilarity-based models.
"""

import torch


class DissimilarityNTXentLoss(torch.nn.Module):
  """
  Normalized Temperature-scaled Cross-Entropy (NT-Xent) loss for contrastive dissimilarity.

  Parameters
  ----------
  temperature : float, optional
      The temperature scaling factor for the softmax operation. Defaults to 0.5.
  """

  def __init__(self, temperature=0.5):
    super(DissimilarityNTXentLoss, self).__init__()
    self.temperature = temperature

  def forward(self, diss, y):
    size = diss.shape[0]

    # Mask for positive samples
    y = torch.cat([y, y], dim=0)
    y1 = torch.tile(y, [size])
    y2 = torch.repeat_interleave(y, size, dim=0)
    pos_mask = torch.reshape(y1 == y2, (size, size))
    pos_mask.fill_diagonal_(False)

    # Mask for negative samples
    neg_mask = (~torch.eye(size, device=diss.device, dtype=bool)).float()

    nominator = torch.sum(pos_mask * torch.exp(diss / self.temperature), dim=1)
    denominator = torch.sum(neg_mask * torch.exp(diss / self.temperature), dim=1)

    loss_partial = -torch.log(nominator / denominator)
    loss = torch.mean(loss_partial)

    return loss


class TripletDissimilarityLoss(torch.nn.Module):
  """
  Triplet dissimilarity loss with a margin.

  Parameters
  ----------
  alpha : float
      The margin for the triplet loss.
  """

  def __init__(self, alpha):
    super(TripletDissimilarityLoss, self).__init__()
    self.alpha = alpha

  def forward(self, positive, negative):
    loss_partial = torch.nn.functional.relu(positive - negative + self.alpha)
    loss = torch.mean(loss_partial)
    return loss
