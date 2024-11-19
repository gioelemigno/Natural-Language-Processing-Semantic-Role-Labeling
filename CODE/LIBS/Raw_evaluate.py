import torch
from typing import *

class Raw_evaluate():
  """
    Object used during the train to evaluate the performance of the model

    It allows to compute the metrics for all the samples in the dataset directly
    avoiding to compute the metrics for each batch and then compute 
    the mean 
  """
  def __init__(self):
    pass

  def update(self, labels: torch.Tensor, predictions: torch.Tensor):
    """
      Update intermedial metrics like `true_positives`
    """
    raise NotImplementedError()

  def evaluate_and_reset(self):
    """
      Compute the final metrics and reset the intermediate ones
    """
    res = self.evaluate()
    self.reset()
    return res
  
  def print(self):
    raise NotImplementedError()

  def reset(self):
    """
      Reset intermediate metrics
    """
    raise NotImplementedError()

  def evaluate(self):
    """
      Compute final metric starting by the intermediate ones
    """
    raise NotImplementedError()
    
  def metrics_names(self):
    """
      get the name of final metrics returned by evaluate()
    """
    raise NotImplementedError()

  def _precision(self, true_positives, false_positives):
    if (true_positives + false_positives) == 0:
      return 0
    else:
      return true_positives / (true_positives + false_positives)

  def _recall(self, true_positives, false_negatives):
    if (true_positives + false_negatives) == 0:
      return 0
    else:
      return true_positives / (true_positives + false_negatives)
      
  def _f1(self, precision, recall):
    if (precision + recall) == 0:
      return 0
    else:
      return 2 * (precision * recall) / (precision + recall)