import torch
from typing import *
from Raw_evaluate import Raw_evaluate


class Raw_evaluate_2(Raw_evaluate):
  """
    Please refer to Raw_evaluate for the documentation
  """

  def __init__(self, null_idx):
    super().__init__()
    self.null_idx = null_idx # PRED_SENSE_MAP.stoi("_")
    self.reset()

  def update(self, labels: torch.Tensor, predictions: torch.Tensor):
    res = self._tensors_raw_evaluate_predicate_disambiguation(labels=labels, predictions=predictions)
    self.pred_disamb_true_positives += res['true_positives']
    self.pred_disamb_false_positives += res['false_positives']
    self.pred_disamb_false_negatives += res['false_negatives']
  
  def print(self):
    print(f"pred_disamb_true_positives={self.pred_disamb_true_positives}")
    print(f"pred_disamb_false_positives={self.pred_disamb_false_positives}")
    print(f"pred_disamb_false_negatives={self.pred_disamb_false_negatives}")

  def reset(self):
    self.pred_disamb_true_positives = 0
    self.pred_disamb_false_positives = 0
    self.pred_disamb_false_negatives = 0

  def evaluate(self):
    # PREDICATE_DISAMBUGUATION
    pred_disamb_precision = self._precision(true_positives=self.pred_disamb_true_positives, 
                                       false_positives=self.pred_disamb_false_positives)

    pred_disamb_recall = self._recall(true_positives=self.pred_disamb_true_positives,
                                 false_negatives=self.pred_disamb_false_negatives)

    pred_disamb_f1 = self._f1(precision=pred_disamb_precision, recall=pred_disamb_recall)

    return {
        'pred_disamb_f1': pred_disamb_f1,
    }
    
  def metrics_names(self):
    return ['pred_disamb_f1']

  def _tensors_raw_evaluate_predicate_disambiguation(self,
                                                    labels: torch.Tensor, 
                                                    predictions: torch.Tensor, 
                                                    ignore_idx=-100):
    null_idx = self.null_idx
    gold = labels
    pred = predictions
    
    mask = (gold != ignore_idx)

    gold_null = (gold == null_idx)
    not_gold_null = torch.logical_not(gold_null)

    pred_null = (pred == null_idx)
    not_pred_null = torch.logical_not(pred_null)

    eq_gold_pred = (gold == pred)
    not_eq_gold_pred = torch.logical_not(eq_gold_pred)

    raw_tp = torch.logical_and(not_gold_null, not_pred_null)
    raw_tp = torch.logical_and(raw_tp, eq_gold_pred)
    clean_tp = torch.logical_and(raw_tp, mask)
    true_positives = torch.sum(clean_tp).item()

    raw_fp_fn = torch.logical_and(not_gold_null, not_pred_null)
    raw_fp_fn = torch.logical_and(raw_fp_fn, not_eq_gold_pred)
    clean_fp_fn = torch.logical_and(raw_fp_fn, mask)
    false_positives_1 = torch.sum(clean_fp_fn).item()
    false_negatives_1 = torch.sum(clean_fp_fn).item()

    raw_fn = torch.logical_and(not_gold_null, pred_null)
    clean_fn = torch.logical_and(raw_fn, mask)
    false_negatives_2 = torch.sum(clean_fn).item()

    raw_fp = torch.logical_and(gold_null, not_pred_null)
    clean_fp = torch.logical_and(raw_fp, mask)
    false_positives_2 = torch.sum(clean_fp).item()

    false_positives = false_positives_1 + false_positives_2
    false_negatives = false_negatives_1 + false_negatives_2

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }