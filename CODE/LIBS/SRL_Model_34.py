from typing import *
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class SRL_Model_34(torch.nn.Module):
    """
      Model to perform Task 3, 4
    """
    def __init__(self, 
                 language_model_name: str, 
                 num_labels: int, 
                 fine_tune_lm: bool = True,
                 positional_embedding_dim: int = 40,
                 lstm_hidden_dim: int = 300,
                 lstm_num_layers: int = 1,
                 lstm_dropout: int = 0.2,
                 classifier_dense_units: int = 300,
                 *args, 
                 **kwargs) -> None:
        super().__init__()

        self.num_labels = num_labels

        # Language model
        self.transformer_model = AutoModel.from_pretrained(language_model_name, output_hidden_states=True)
        if not fine_tune_lm:
            for param in self.transformer_model.parameters():
                param.requires_grad = False
        self.dropout = torch.nn.Dropout(0.2)
        
        # Positional embedding
        '''
          input:
              predicate_idx     # index of the predicate in the original sentence
              predicate_sense   # sense of the predicate in the sentence
              word_ids          # Mapping original sentence to `input_ids`
          output:
              positional_embedding 

          e.g.
            original_sentence = "Obama", "went", "to", "Italy" 
            srl_tags = ('agent', '_', '_', 'destination')
            input_ids = [CLS], "Obama", "went", "to", "Italy", [SEP], "went", [SEP]

            predicate_idx = 1
            predicate_sense = GO-FORWARD
            word_ids = [-99, 0, 1, 2, 3, -99, -99, -99]

            For each component of `word_ids` a positional_embedding is generated

            Example:
              predicate_sense = [ G-F, G-F, G-F, G-F, G-F, G-F, G-F, G-F,
              predicate_idx =   [   1,  1,   1,   1,   1,    1,   1,   1]
              word_ids =        [ -99,  0,   1,   2,   3,  -99, -99, -99]

              positional_embedding = Tensor 8 x pos_emebedding_dim
        '''
        self.positional_embedding_dim = positional_embedding_dim
        self.positional_embedding = nn.Linear(3, self.positional_embedding_dim, bias=False)
        

        # BiLSTM
        '''
          BiLSTM takes as input:
            - Language model output
            - POS Tag
            - Positional embedding
            - Predicate sense one hot alligned 
              E.g.
                predicate_sense_one_hot_alligned:[  _, _, G-F, _, _,   _,   _,   _]
                                      word_ids = [-99, 0,   1, 2, 3, -99, -99, -99]

        '''
        input_lstm_dim = self.transformer_model.config.hidden_size 
        input_lstm_dim += 1 
        input_lstm_dim += self.positional_embedding_dim 
        input_lstm_dim += 1
        self.lstm = nn.LSTM(input_lstm_dim, lstm_hidden_dim, 
                            bidirectional=True,
                            num_layers=lstm_num_layers, 
                            dropout = lstm_dropout if lstm_num_layers > 1 else 0,
                            batch_first=True)
        lstm_output_dim = lstm_hidden_dim * 2
        
        # CLASSIFIER
        '''
          Final multi-layer classifier
        '''
        input_classifier_dim = lstm_output_dim  
        self.classifier = torch.nn.Sequential(
            nn.Linear(input_classifier_dim, classifier_dense_units, bias=False),
            nn.ReLU(),
            nn.Linear(classifier_dense_units, num_labels),
        )
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
                predicate_sense=None,
                word_ids=None,
                predicate_sense_one_hot=None,
                *args,
                **kwargs) -> torch.Tensor:

        # prepare input for the language model
        model_kwargs = {
          "input_ids": input_ids, 
          "attention_mask": attention_mask
        }
        # not every model supports token_type_ids
        if not token_type_ids is None:
          model_kwargs["token_type_ids"] = token_type_ids

        # sum of the last four hidden layers
        transformers_outputs = self.transformer_model(**model_kwargs)
        transformers_outputs_sum = torch.stack(transformers_outputs.hidden_states[-4:], dim=0).sum(dim=0)
        transformers_outputs_sum = self.dropout(transformers_outputs_sum)
        
        # build positional embedding
        inputs_ids_len = input_ids.size()[-1] 
        #
        pred_idx = predicate_idx.unsqueeze(dim=-1).expand(-1, inputs_ids_len).unsqueeze(dim=-1)
        pred_sense = predicate_sense.unsqueeze(dim=-1).expand(-1, inputs_ids_len).unsqueeze(dim=-1)
        w_ids = word_ids.unsqueeze(dim=-1)
        #
        in_emb = torch.cat((pred_idx, pred_sense, w_ids), dim=-1)
        in_emb = in_emb.type(torch.float) # cast
        #
        positional_emb = self.positional_embedding(in_emb)

        # unsqueeze POS tag
        pos_ids_3D = torch.unsqueeze(pos_ids, dim=-1)

        # pred_sense_one_hot 
        predicate_sense_one_hot_alligned_3D = torch.unsqueeze(predicate_sense_one_hot, dim=-1)
        
        # build lstm input
        input_lstm = torch.cat((transformers_outputs_sum, pos_ids_3D, positional_emb, predicate_sense_one_hot_alligned_3D), dim=-1)

        o, (h, c) = self.lstm(input_lstm)

        input_classifier = o  
        logits = self.classifier(input_classifier)

        output = {"logits": logits}

        if compute_predictions:
            predictions = logits.argmax(dim=-1)
            output["predictions"] = predictions

        if compute_loss and labels is not None:
            output["loss"] = self.compute_loss(logits, labels)

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


def trainer_forward_fn_34(batch: torch.Tensor, model:SRL_Model_34,  device: str) -> None:
  """
    Function used to give a batch in input to the model during the training
        # Parameters
          `batch`, `torch.Tensor`, required
            Batch of samples to give as input to the model

          `model`, `SRL_Model_34`, required
            Model

          `device`, `str`, required
            Device used by the model

  """
  batch = {k: v.to(device) for k, v in batch.items()}
  labels = batch['srl_ids']
  batch_out = model(**batch, compute_predictions=True, compute_loss=True, labels=labels)
  batch_out['labels'] = labels
  return batch_out
