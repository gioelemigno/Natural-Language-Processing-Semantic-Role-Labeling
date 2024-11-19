from typing import *
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

from SRL_Model_2 import SRL_Model_2
from SRL_Model_34 import SRL_Model_34

class SRL_Model_234(torch.nn.Module):
    """
      Model to perform Task 2, 3, 4
    """
    def __init__(self, 
                  hparams_SRL_2,
                  hparams_SRL_34,
                  null_tag_sense_idx,
                  *args, 
                  **kwargs) -> None:
        super().__init__()

        self.null_tag_sense_idx = null_tag_sense_idx
        self.srl_model_2 = SRL_Model_2(**hparams_SRL_2)
        self.srl_model_34 = SRL_Model_34(**hparams_SRL_34)
        self.eval()

    def forward(self,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                labels: torch.Tensor = None,
                compute_predictions: bool = False,
                compute_loss: bool = True,
                pos_ids: torch.Tensor = None,
                predicate_idx=None,
                word_ids=None,
                predicate_idx_one_hot_alligned=None,
                *args,
                **kwargs,
            ) -> torch.Tensor:

        # build input model srl_2
        srl_2_model_kwargs = {
          "input_ids": input_ids, 
          "attention_mask": attention_mask,
          'compute_predictions': True,
          'pos_ids': pos_ids,
          'predicate_idx': predicate_idx,
          'word_ids': word_ids,
          'predicate_idx_one_hot_alligned': predicate_idx_one_hot_alligned,
        }
        
        out_srl_model_2 = self.srl_model_2(**srl_2_model_kwargs)

        predicate_sense_one_hot = out_srl_model_2['predictions']

        # CLEAN OUTPUT OF SRL_MODEL_2 AND GET PRED_SENSE
        # select only one predicate sense per sentence
        predicate_sense_one_hot_force_only_one = predicate_sense_one_hot * (predicate_idx_one_hot_alligned == 1)
        #
        #mantain null index like -100 or -99
        predicate_sense_one_hot_force_only_one += predicate_idx_one_hot_alligned * (predicate_idx_one_hot_alligned < 0) 
        #
        # create a selection mask
        mask = (predicate_idx_one_hot_alligned == 1)
        where_mask_is_all_false = mask.sum(dim=1)==0
        #
        ids_where_mask_is_all_false = torch.nonzero(where_mask_is_all_false)
        ids_where_mask_is_all_false = ids_where_mask_is_all_false.type(dtype=torch.long)
        #
        # we will set to true the first element of all row that contain only False
        # this allow us to select -100 when a predicate sense is not present in the
        # sentcnce. Then we will substitute it to null_tag_idx
        zeros = torch.zeros(ids_where_mask_is_all_false.size(), dtype=torch.long)
        #
        mask[ids_where_mask_is_all_false, zeros] = True
        #
        # apply the mask
        predicate_sense_to_clean = torch.masked_select(predicate_sense_one_hot_force_only_one, mask)
        #
        # substitute -100 with null_tag_idx
        ids_to_clean = torch.nonzero(predicate_sense_to_clean < 0)
        predicate_sense_to_clean[ids_to_clean] = self.null_tag_sense_idx
        predicate_sense = predicate_sense_to_clean

        
        # build input model
        srl_34_model_kwargs = {
          "input_ids": input_ids, 
          "attention_mask": attention_mask,
          'compute_predictions': compute_predictions,
          'compute_loss': compute_loss,
          'pos_ids': pos_ids,
          'predicate_idx': predicate_idx,
          'word_ids': word_ids,
          'predicate_idx_one_hot_alligned': predicate_idx_one_hot_alligned,
          'predicate_sense_one_hot': predicate_sense_one_hot_force_only_one,
          'predicate_idx': predicate_idx,
          'labels': labels,
          'predicate_sense': predicate_sense
        }

        out_srl_model_34 = self.srl_model_34(**srl_34_model_kwargs)

        output = out_srl_model_34
        return output

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss of the model.
          # Parameters
              `logits`: `torch.Tensor`, required
                  The logits of the model.

              `labels`:`torch.Tensor`, required
                  Labels, (a.k.a ground truth or gold)
          Returns:
              obj:`torch.Tensor`: The loss of the model.
        """
        return F.cross_entropy(
            logits.view(-1, self.num_labels),
            labels.view(-1),
            ignore_index=-100,
        )
      
    def save_model(self, filepath: str) -> None:
      """
        Save model's weights on file at 'filepath'
        # Parameters
            `filepath`, `str`, required
              Path of the file in which save model's weights 
      """
      torch.save(self.state_dict(), filepath)  # save the model state
    
    def load_model(self, filepath:str, device:str) -> None:
      """
        Load model's weights from a file at 'filepath'
        # Parameters
            `filepath`, `str`, required
              Path of the file in which model's weights are stored

            `device`, `str`, required
              Device used by the model
      """
      # https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
      self.load_state_dict(torch.load(filepath, map_location=torch.device(device)))


def trainer_forward_fn_234(batch, model, device):
  """
    Function used to give a batch in input to the model during the training
        # Parameters
          `batch`, `torch.Tensor`, required
            Batch of samples to give as input to the model

          `model`, `SRL_Model_234`, required
            Model

          `device`, `str`, required
            Device used by the model

  """
  batch = {k: v.to(device) for k, v in batch.items()}
  labels = batch['srl_ids']
  batch_out = model(**batch, compute_predictions=True, compute_loss=True, labels=labels)
  batch_out['labels'] = labels
  return batch_out
