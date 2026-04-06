"""Open-set recognition scoring and evaluation utilities.

Provides scoring functions (MSP, MLS, MDS) and evaluation metrics
(closed-set accuracy, AUROC, OpenAUC) for open-set recognition tasks.
"""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler


def compute_msp(probs):
  """Maximum Softmax Probability.

  Parameters
  ----------
  probs : np.ndarray
      Predicted probabilities of shape (n_samples, n_classes).

  Returns
  -------
  np.ndarray
      The maximum softmax probability for each sample.
  """
  return probs.max(axis=1)


def compute_mls(logits):
  """Maximum Logit Score.

  Parameters
  ----------
  logits : np.ndarray
      Predicted logits of shape (n_samples, n_classes).

  Returns
  -------
  np.ndarray
      The maximum logit score for each sample.
  """
  return logits.max(axis=1)


def compute_mds(X_train, X_test, Y_test=None, inverted=False):
  """Minimum Dissimilarity Score.

  Parameters
  ----------
  X_train : np.ndarray
      Training data of shape (n_samples, n_features).
  X_test : np.ndarray
      Test data of shape (n_samples, n_features).
  Y_test : np.ndarray, optional
      Class labels for the test data. Used for dissimilarity vector
      representation to reshape the test data. Defaults to None.
  inverted : bool, optional
      If True, inverts the scores (useful for contrastive loss where
      higher scores indicate more dissimilarity). Defaults to False.

  Returns
  -------
  np.ndarray
      The minimum dissimilarity score for each test sample.
  """

  scaler = MinMaxScaler(clip=True).fit(X_train)
  X_test_scaled = scaler.transform(X_test)

  # Reshape for dissimilarity vector representation
  if Y_test is not None:
    X_test_scaled = np.reshape(X_test, (Y_test.shape[0], -1))

  if inverted:
    return np.min(1 - X_test_scaled, axis=1)
  return np.min(X_test_scaled, axis=1)


def closed_accuracy(preds, y_true, mask):
  """Closed-set accuracy on masked (known) samples.

  Parameters
  ----------
  preds : np.ndarray
      Predicted class labels.
  y_true : np.ndarray
      True class labels.
  mask : np.ndarray
      Boolean mask indicating known samples.

  Returns
  -------
  float
      Closed-set accuracy for the known samples.
  """
  return accuracy_score(y_true[mask], preds[mask])


def open_auroc(scores, known_mask, unknown_mask):
  """AUROC for open-set detection.

  Parameters
  ----------
  scores : np.ndarray
      Predicted scores for the test data.
  known_mask : np.ndarray
      Boolean mask for known samples.
  unknown_mask : np.ndarray
      Boolean mask for unknown samples.

  Returns
  -------
  float
      The AUROC score for open-set detection.
  """
  y = np.concatenate([np.zeros(np.sum(known_mask)), np.ones(np.sum(unknown_mask))])
  s = np.concatenate([scores[known_mask], scores[unknown_mask]])
  return roc_auc_score(y, s)


def openauc(scores, predictions, y_true, known_mask, unknown_mask):
  """Compute OpenAUC.

  Parameters
  ----------
  scores : np.ndarray
      Predicted scores for the test data.
  predictions : np.ndarray
      Predicted class labels.
  y_true : np.ndarray
      True class labels.
  known_mask : np.ndarray
      Boolean mask for known samples.
  unknown_mask : np.ndarray
      Boolean mask for unknown samples.

  Returns
  -------
  float
      The OpenAUC score.
  """

  x1 = scores[known_mask].tolist()
  x2 = scores[unknown_mask].tolist()
  p1 = predictions[known_mask]
  l1 = y_true[known_mask]

  correct = (p1 == l1).tolist()

  # Misclassified known samples are forced to look like the worst unknown
  m_x2 = max(x2) + 1e-5

  y_score = [value if hit else m_x2 for value, hit in zip(x1, correct)] + x2
  y_labels = [0]*len(x1) + [1]*len(x2)

  return roc_auc_score(y_labels, y_score)
