from typing import *
from pipeline_utils import one_hot_encoding, one_hot_encoding_to_idx, get_predicates_indexes, get_data, All_in_RAM_Dataset, get_allgned_idx, to_batch, add_pos_tags
from torch.utils.data import Dataset, DataLoader
from Mapping_string_int import Mapping_string_int
import torch

from SRL_Model_234 import SRL_Model_234

def load_sample_234(id_sample: Any, 
                  sample_in: Dict[str, Any]) -> Dict[str, List[Any]]:
  """
  Used to load a single sample from JSON dataset to perform TASK 34
  # Parameters
  
    `id_sample`: `Any`, required
        ID of the sample to load

    `sample_in`: `Dict[str, Any]`, required
        Single sample loaded from the JSON dataset file
        NOTE: A sample that contains multiple predicates is splitted
              in different samples, one for each predicate

  # Returns 
  An output dictionary consisting of:

    `ids` : `List[Any]`
      List samples IDs

    `xs` : `List[Dict[str, Any]]`
      List of samples, each of them is a dictionary containing 
      model input and output
      
      Each dictionary contains:
        `words`: `List[str]`
          Input sentence splitted in words

        `predicate_idx`: `int`
          index of the only considered predicate in the sentence
          if there is no predicate, then it has a negative value

        `predicate_idx_one_hot`: `List[int]`
          one-hot encoding representing the position of the considered predicate 
            e.g. [0, 0, 1, 0, 0, 0] # predicate at index 2
                 [0, 0, 0, 0] # no predicate
  
        `srl_tags`: `List[str]`
          Semantic roles of the considered predicate
          if no predicate, then it is a list filled of '_'
      

    `ys` : `List[Dict[str, Any]]`
      List of samples, each of them is a dictionary 
      containing only model output (gold)

      Each dictionary contains:
        `srl_tags`: `List[str]`
          Semantic roles of the considered predicate
          if no predicate, then it is a list filled of '_'
  """  

  data = {
      'xs': [],
      'ys': [],
      'ids': []
  }

  res = get_predicates_indexes(sample_in['predicates'])
  pred_ids = res['pred_ids']
  
  if len(pred_ids) > 0:
    #
    # no predicate sense
    #
    for i, pred_idx in enumerate(pred_ids):
      x = {
          'words': sample_in['words'],
          'predicate_idx': pred_ids[i],
          'predicate_idx_one_hot': one_hot_encoding(index=pred_ids[i], 
                                                    length=len(sample_in['words']))
      }
      #
      if 'roles' in sample_in.keys():
        _keys = list(sample_in['roles'].keys())
        _keys_type = type(_keys[0])
        if _keys_type == int:
          _key = pred_idx
        elif _keys_type == str:
          _key = str(pred_idx)
        else:
          raise Exception("Internal error: unknown type of keys in 'roles")
        #
        x['srl_tags'] = sample_in['roles'][_key]
        #
        y = x['srl_tags']
        data['ys'].append(y)
        #
      #
      data['xs'].append(x)
      data['ids'].append(id_sample)
  else: # no predicates in the sentence
    x = {
      'words': sample_in['words'],
      'srl_tags' : ['_']*len(sample_in['words']),
       #
      'predicate_idx': -99,
      'predicate_idx_one_hot': [0]*len(sample_in['words']),
    }
    #
    y = x['srl_tags']
    data['xs'].append(x)
    data['ids'].append(id_sample)
    data['ys'].append(y)
  return data


def collate_fn_args_234(batch: List[Dict[str, Dict[str, Any]]], 
                      tokenizer: object, 
                      srl_map: Mapping_string_int, 
                      pred_sense_map: Mapping_string_int, 
                      pos_tag_map: Mapping_string_int) -> Dict[str, torch.Tensor]:
    """
    Collate function used by Dataloader to process a batch of sample 
    before the model

    # Parameters
      `batch`: `List[Dict[str, Dict[str, Any]]]`, required
          Batch of samples given in input by Dataloader

      `tokenizer`: `object`, required
          Transformer tokenizer

      `srl_map`: `Mapping_string_int`, required
          Mapper for SRL TAGS

      `pred_sense_map`: `Mapping_string_int`, required
          Mapper for predicate sense tags

      `pos_tag_map`: `Mapping_string_int`, required
          Mapper for POS tags

    # Returns 
      `Dict[str, torch.Tensor]`. 
        Returns a dictionary with all input/output batches for the model:

          `input_ids`: `List[List[int]]`, always
              Input sentences tokenized. (output tokenizer)

          `attention_mask`: `List[List[int]]`, always
              Attention mask. (output tokenizer)

          `word_ids`: `List[List[int]]`, always
              Each element at position `i` of a item, indicates the index 
              in the original sentence of the item to which `input_ids[i]` 
              refers to. (output tokenizer)

          `pos_ids`: `List[List[int]]`, always
              POS Tags converted to `int` and alligned to `input_ids`

          `predicate_idx`: `List[int]`, always
              Position of the predicate in the original sentence

          `predicate_idx_one_hot`: `List[int]`, always
            Position of the predicate in the original sentence, indicates 
            using a one-hot 0/1 encoding

          `predicate_idx_one_hot_alligned`: `List[List[int]]`, always
            `predicate_idx_one_hot` alligned to the tokenized sentence

          ---

          `srl_ids`: `List[List[int]]`, if 'srl_tags' in batch[0].keys()
            SRL Tags converted to int and alligned to `input_ids`

    """
    # BUILD FLAG
    use_srl_tags = False
    if 'srl_tags' in batch[0]['x'].keys():
      use_srl_tags = True

    # TOKENIZATION TEXT WITH PREDICATE AS SECOND INPUT
    ## Build input for the tokenizer
    seq_1 = list()
    seq_2 = list()
    for item in batch:
      words = item['x']['words']
      pred_idx = item['x']['predicate_idx']
      if pred_idx < 0:
        predicate = '-'
      else:
        predicate = words[pred_idx]
      
      seq_1.append(words)
      seq_2.append([predicate])
      
    batch_out = tokenizer(
        seq_1, seq_2,
        return_tensors="pt",
        padding=True,
        # We use this argument because the texts in our dataset are lists of words.
        is_split_into_words=True,
    )

    # DEFINITION OF ADDITIONAL BATCHS TO ADD TO batch_out
    batch_word_ids = list()
    batch_pos_ids_alligned = list() # POS Tags
    batch_predicate_idx = list()
    #
    batch_predicate_idx_one_hot = list()
    batch_predicate_idx_one_hot_alligned = list()
    
    if use_srl_tags:
      batch_srl_ids_alligned = list()
    
    #
    for i, item in enumerate(batch):
      #
      # get mapping word <-> output_tokenizer
      word_ids = batch_out.word_ids(batch_index=i)
      #
      batch_predicate_idx.append(item['x']['predicate_idx'])
      predicate_idx_one_hot = item['x']['predicate_idx_one_hot']
      batch_predicate_idx_one_hot.append(predicate_idx_one_hot)
      #
      if use_srl_tags:
        srl_tags = item['x']['srl_tags']
        srl_ids_alligned = list()
      #
      pos_tags = item['x']['pos_tags']
      pos_ids_alligned = list()
      #
      predicate_idx_one_hot_alligned = list()
      #
      previous_word_idx = None
      none_counter = 2
      #ic(word_ids)
      for word_idx in word_ids:
        if word_idx is None and none_counter > 0:
          none_counter -= 1

        if use_srl_tags:
          srl_idx = get_allgned_idx(srl_tags, 
                                    word_idx, 
                                    previous_word_idx, 
                                    none_counter, 
                                    srl_map)
          srl_ids_alligned.append(srl_idx)
        
        pred_idx_one_hot_alligned_idx = get_allgned_idx(predicate_idx_one_hot, 
                                                        word_idx, 
                                                        previous_word_idx, 
                                                        none_counter, 
                                                        tag_map=None)
        predicate_idx_one_hot_alligned.append(pred_idx_one_hot_alligned_idx)

        pos_idx = get_allgned_idx(pos_tags, 
                                  word_idx, 
                                  previous_word_idx, 
                                  none_counter, 
                                  pos_tag_map)
        pos_ids_alligned.append(pos_idx)
        #
        previous_word_idx = word_idx
      #
      if use_srl_tags:
        batch_srl_ids_alligned.append(srl_ids_alligned)
      #
      batch_pos_ids_alligned.append(pos_ids_alligned)
      #
      batch_predicate_idx_one_hot_alligned.append(predicate_idx_one_hot_alligned)
      #
      # replace None with -99 to be compatible with tensor requirements
      word_ids = [-99 if x is None else x for x in word_ids]
      batch_word_ids.append(word_ids)
    
    batch_out["word_ids"] = torch.as_tensor(batch_word_ids)
    batch_out['predicate_idx'] = torch.as_tensor(batch_predicate_idx)
    batch_out['pos_ids'] = torch.as_tensor(batch_pos_ids_alligned)
    #
    batch_predicate_idx_one_hot = to_batch(batch_predicate_idx_one_hot)
    batch_out['predicate_idx_one_hot'] = torch.as_tensor(batch_predicate_idx_one_hot)
    batch_out['predicate_idx_one_hot_alligned'] = torch.as_tensor(batch_predicate_idx_one_hot_alligned)
    #
    if use_srl_tags:
      batch_srl_ids_alligned = to_batch(batch_srl_ids_alligned, pad=-100)
      batch_out["srl_ids"] = torch.as_tensor(batch_srl_ids_alligned)
    #
    #
    return batch_out



class End_to_end_predictor_SRL_234():
  """
  Class used to create a object able to perform a complete prediction
  of a sample (end to end) (i.e. string input -> string input)

  # Parameters init
      `model`: `SRL_Model_234`, required
        PyTorch model

      `device`: `str`, required
        Device to use (where load model and tensors)

      `srl_map`: `Mapping_string_int`, required
        Mapper for SRL Tags

      `pred_sense_map`: `Mapping_string_int`, required
        Mapper for predicate senses

      `load_sample_fn`: `Dict[str, List]`, required
        Already loaded data.
        If not None, `file_in_path` and `get_data_func` are ignored

      `collate_fn`: `Callable`, required
        Collate function used by Dataloader to process a batch of sample 
        before the model

      `pre_processing_funcs`: `List[Callable]`, required
        Function used to add features to the samples in `data` loaded with 
        `load_sample_fn()`
  """

  def __init__(self, 
               model: SRL_Model_234, 
               srl_map: Mapping_string_int, 
               pred_sense_map: Mapping_string_int,
               pre_processing_funcs: List[Callable],
               device: str,
               load_sample_fn: Callable,
               collate_fn: Callable):
    self.model = model

    self.srl_map = srl_map
    self.pred_sense_map = pred_sense_map
    
    self.device = device

    self.load_sample_fn = load_sample_fn
    self.collate_fn = collate_fn

    self.pre_processing_funcs = pre_processing_funcs

  def _clean_logits(self, logits, word_ids):
    """
      Function used to remove logits associated to special token 
      of the transformer, namely those indicated by`word_ids' with -99 (None) 
    """
    clean_logits = []
    #
    #ic(logits)
    #ic(word_ids)
    #
    previous_idx = None 
    none_counter = 2
    for i, word_idx in enumerate(word_ids):
      if none_counter == 0:
        break
      elif word_idx == -99:
        none_counter -= 1
      elif word_idx != previous_idx:
        clean_logits.append(logits[i])
        previous_idx = word_idx
      else:
        continue
    #
    clean_logits = torch.stack(clean_logits, dim=0)
    #ic(clean_logits)
    
    return clean_logits
  
  def predict(self, 
              sentence: List[str]) -> Dict[str, List[str]]:
    """
    End-to-end prediction

    # Parameters
        `sentence`: `List[str]`, required
          Input sentence

    # Return
      `Dict[List[str]]` prediction
    """
    res = {}
    data = self.load_sample_fn(id_sample=None, sample_in=sentence)
    dataset = All_in_RAM_Dataset(data=data, pre_processing_funcs=self.pre_processing_funcs)
    dataloader = DataLoader(dataset, 
                              batch_size=len(data['xs']), # :')
                              collate_fn=self.collate_fn,
                              num_workers=0,
                              shuffle=False)
    for batch in dataloader:
      continue
      raise Exception("Internal Error: Sentence splitted in more than one batch")


    batch = batch.to(self.device)
    self.model.to(self.device)
    self.model.eval()
    
    with torch.no_grad(): # avoid gradient computation (avoid waste of time)
      out = self.model(**batch, compute_predictions=True)
      logits = out['logits']
      roles = dict()
      for i, pred_idx in enumerate(batch['predicate_idx']):
        if pred_idx < 0: # no predicates in the sentence, so no roles
          continue       # roles = {}
        cleaned_logits = self._clean_logits(logits[i], batch['word_ids'][i])
        cleaned_preds = torch.argmax(cleaned_logits, dim=-1)
        #
        preds_tags = [self.srl_map.itos(idx) for idx in cleaned_preds]
        roles[pred_idx.item()] = preds_tags

      res['roles'] = roles
    return res
