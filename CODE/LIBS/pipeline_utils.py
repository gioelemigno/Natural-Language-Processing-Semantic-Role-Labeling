from torch.utils.data import Dataset, DataLoader
from typing import *
from Mapping_string_int import Mapping_string_int
from POS_Tagger import POS_Tagger
import json

def get_predicates_indexes(predicates: List[Any], 
                           blanks: List[Any] = ['_', 0]
                           ) -> Dict[str, List[Any]]:

  """
  Used to extract predicate positions (indexes) and senses
  # Parameters
  
  `predicates`: List[Any], required
      List of predicate senses or predicate flags (strings or 0-1)

      e.g.
        List[str] -> ['_', '_', 'ASK_REQUEST', '_', '_', 'BENEFIT_EXPLOIT']
        List[int] -> [0, 0, 1, 0, 0, 1]

  `blanks`: List[Any], optional, (default = ['_', 0])
      List of predicates (strings or 0-1)

      e.g.
        List[str] -> ['_', '_', 'ASK_REQUEST', '_', '_', 'BENEFIT_EXPLOIT']
        List[int] -> [0, 0, 1, 0, 0, 1]

  # Returns 
  An output dictionary consisting of:

    if `predicates` is List[str]:
      `pred_ids` : `List[int]`
        List of predicate indexes  
      `pred_senses` : `List[int]`
        List of predicate sense  

    elif `predicates` is List[int]:
      `pred_ids` : `List[int]`
        List of predicate indexes  

  """  

  pred_ids = list()
  pred_senses = list()
  for i, elem in enumerate(predicates):
    if not elem in blanks:
      pred_ids.append(i)
      if type(elem) == str:
        pred_senses.append(elem)
  res = {'pred_ids': pred_ids}
  if len(pred_senses) > 0:
    #ic(len(pred_senses)==len(pred_ids))
    assert(len(pred_senses)==len(pred_ids))
    res['pred_senses'] = pred_senses
  return res

def get_data(dataset_path: str, 
             load_sample_fn: Callable
             ) -> Dict[str, List[Any]]:
  """
  Used to load data from a JSON dataset file.
  Each sample in the JSON file is loaded using `load_sample_fn` function 
  
  # Parameters
  
  `dataset_path`: str, required
      Path of the dataset in JSON format

  `load_sample_fn`: Callable, required
      Function used to load a single sample
      NOTE: A sample that contains multiple predicates is splitted
            in different samples, one for each predicate

  # Returns 
  An output dictionary consisting of:

    `ids` : `List[Any]`
      List samples IDs

    `xs` : `List[Dict[str, Any]]`
      List of samples, each of them is a dictionary containing 
      model input and output

    `ys` : `List[Dict[str, Any]]`
      List of samples, each of them is a dictionary 
      containing only model output (gold)

  """  

  data = {
            'ids':[],
            'xs':[],
            'ys':[]
          }
  with open(dataset_path, 'r') as f:
    loaded = json.load(f)

  for i, key in enumerate(loaded.keys()):
    sample_in = loaded[key]
    #
    data_sample = load_sample_fn(id_sample=key, sample_in=sample_in)
    #
    data['xs'] += data_sample['xs']
    data['ys'] += data_sample['ys']
    data['ids'] += data_sample['ids']
    #
  return data


def get_allgned_idx(tags: List[Any], 
                    word_idx: int, 
                    previous_word_idx: int, 
                    none_counter: int, 
                    tag_map: Optional[Mapping_string_int]=None) -> int:
  """
  Function used in `collate_fn` to allign elements in `tags` that refer to 
  original sentence, to the output of the tokenizer (`input_ids`) 

  # Parameters
  
    `tags`: `List[Any]`, required
        List of tags (str or int)
        e.g. pos_tags
              ['DT', 'NNP', 'VBZ', 'IN', 'DT', 'NNP', 'NN', 'NN', 'JJR', 'NN', 
              'TO', 'VBG', 'DT', 'NNS', 'IN', 'DT', 'NN', 'NN', 'NNS', ',', 
              'NN', 'CC', 'VBD', 'NNS', 'IN', 'JJ', 'NNS', '.']

    `word_idx`: `int`, required
        Current word_idx from `word_ids` of tokenizer output
          if non-negative indicates the index of the word in the original
          sentence to which the tokenizer output (element of `input_ids`) 
          refers to.
          if None indicates that the tokenizer output refer to a special
          token e.g. (101)[CLS], (102)[SEP]
        e.g. `word_ids`
          'words': ['The', 'Committee', 'recommends', 'that', 'the', 'State', 'party', 'pay', 'more',
                    'attention', 'to', 'sensitizing', 'the', 'members', 'of', 'the', 'law',
                    'enforcement', 'agencies', ',', 'security', 'and', 'armed', 'forces', 'about',
                    'human', 'rights', '.']}}

          'input_ids': tensor([  101,  1996,  2837, 26021,  2008,  1996,  2110,  2283,  3477,  2062,
                            3086,  2000, 12411, 28032,  6026,  1996,  2372,  1997,  1996,  2375,
                            7285,  6736,  1010,  3036,  1998,  4273,  2749,  2055,  2529,  2916,
                            1012,   102, 26021,   102]  

          'word_ids': [None,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  11,
                              11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
                              25,  26,  27, None,   0, None]

    `previous_word_idx`: `int`, required
        Single word_idx from `word_ids` of tokenizer output, previous word index

    `none_counter`: `int`, required
        Counter used to count the occurrences of None. 
        After two Nones in `word_ids` the original sentence is ended

    `tag_map`: `Mapping_string_int`, Optional (default=None)
        Mapping_string_int used by `tags`, it used to convert a tag into integer
      
  # Returns 
    `int`. Returns the converted element of `tags` that correspond to `word_idx`

  """  

  res = None
  # Special tokens have a word id that is None. 
  # We set the label to -100 so they are automatically
  # ignored in the loss function.
  if none_counter == 0: #ended first sequence
    res = -100
  elif word_idx is None: # special char of transformer
    res = -100
  # We set the label for the first token of each word.
  elif word_idx != previous_word_idx:
    if tag_map:
      res = tag_map.stoi(tags[word_idx])
    else:
      res = tags[word_idx]
  # For the other tokens in a word, 
  # we set the label to -100 so they are automatically
  # ignored in the loss function.
  else:
    res = -100
  return res     


def to_batch(items: List[List[Any]], 
             pad: Optional[Any]=-100) -> List[List[Any]]:
  """
  Transform a list of samples into a batch-list by padding samples 

  # Parameters
  
    `items`: `List[List[Any]]`, required
        List of samples, each sample is a list with arbitrary length

    `pad`: `Any`, optional (default=-100)
        Element used to pad all samples

  # Returns 
    `List[List[Any]]`. 
      Returns the list of sample `items` padded with `pad`, 
      ready to transformed in batch-Tensor

  """  
  batch_max_length = len(max(items, key=len))
  res = [l + ([pad] * abs(batch_max_length - len(l))) for l in items]
  return res 



class All_in_RAM_Dataset(Dataset):
  """
  Standard dataset that load all sample in memory
  # Parameters init
      `file_in_path`: `str`, Optional (default=None)
        Path of JSON dataset to load using `get_data_func`

      `get_data_func`: `Callable`, Optional (default=None)
        Function used to load dataset saved in `file_in_path`

      `data`: `Dict[str, List]`, Optional (default=None)
        Already loaded data.
        If not None, `file_in_path` and `get_data_func` are ignored

      `pre_processing_funcs`: `List[Callable]`, Optional (default=[]])
        Function used to add features to the samples in `data`
  """
  def __init__(self, 
               file_in_path: Optional[str]=None, 
               get_data_func: Callable=None, 
               data: Optional[Dict]=None, 
               pre_processing_funcs: List[Callable]=[]):
    if data is None:
      self.data = get_data_func(file_in_path)
    else:
      self.data = data
    #
    for func in pre_processing_funcs:
      self.data = func(self.data)

    
  def __len__(self):
    return len(self.data['xs'])
      
  def __getitem__(self, idx):
    item = {
        'x': self.data['xs'][idx],
        #'y': self.data['ys'][idx],
        'id': self.data['ids'][idx]
    }
    return item


def one_hot_encoding(index: int, 
                     length: int, 
                     one: Optional[Any]=1, 
                     zero: Optional[Any]=0) -> List[Any]:
  """
    Create a one-hot encoding

    # Parameters
        `index`: `int`, Required
        Index of the element to set to `one`

        `length`: `int`, Required
        Length of the output 

        `one`: `Any`, Optionale (default=1)
        Element to use as one 

        `zero`: `Any`, Optionale (default=0)
        Element to use as zero

  """
  l = [zero] * length 
  l[index] = one
  return l

def one_hot_encoding_to_idx(one_hot: List[Any], 
                            zero: Optional[Any]=0) -> int:
  """
    Reverse a one-hot encoding (get an index)

    # Parameters
        `one_hot`: `List[Any]`, Required
        one-hot encoding

        `zero`: `Any`, Optionale (default=0)
        Element to used as zero

  """
  not_zero = list(filter(lambda x: x != zero, one_hot))
  if len(not_zero) != 1:
    raise Exception("invalid one-hot encoding")
  return one_hot.index(not_zero[0])


def add_pos_tags(data: List[Dict[str, Dict[str, Any]]], 
                 pos_tagger: POS_Tagger):

  """
  Add POS Tags to the data loaded using `get_data()`

  # Parameters
    `data`: `List[Dict[str, Dict[str, Any]]]`, required
        Output of `get_data()`

    `pos_tagger`: `POS_Tagger`, required
        Transformer tokenizer

  # Returns 
    `List[Dict[str, Dict[str, Any]]]`. 
      `data` augmented with POS Tags

  """
  for x in data['xs']:
    x['pos_tags'] = pos_tagger.pos_tagging(x['words'])
    if len(x['pos_tags']) != len(x['words']):
      print("WARNING: Pos tags and words have different lenghts")
  return data 
