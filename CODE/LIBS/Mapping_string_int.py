from typing import *
import random

class Mapping_string_int():
  """
  Manager of string <-> int mapping
  # Parameters init
      `counter`: `Counter`, required
        Counter used to create the list of known labels

      `min_freq`: `int`, Optional (default=None)
         ignore those string with a frequency lower than min_freq
         NOTE: To avoid exceptions calling `self.stoi()` is required to 
               set to true `self.allow_unknown_tag` 

      `special_strings`: 'List[str]', Optional (default=None)
        List of special strings do not contained in `counter` 
        to add to the list of known labels

      `shuffle`: 'bool', Optional (default=True)
        shuffle the list of labels

      `allow_unknown_tag`: 'bool', Optional (default=False)
        An placeholder tag ('<UNK_TAG>') is added to the list of known strings,
        when an unknown string is given in input to `self.stoi()` then the index
        of the placeholder is returned instead of Raise an exception   

  # Attributes
    In addition to parameters of init:

      `skipped_words`: `List[str]`
        List of skipped string due to their frequency below `self.min_freq`

      `list_string`: `List[str]`
        List of known labels, it can contain the placeholder tag ('<UNK_TAG>')
  """
  def __init__(self, 
               counter: Counter,
               min_freq: Optional[int]=None,
               special_string: Optional[List[str]]=None,
               shuffle: Optional[bool]=True,
               allow_unknown_tag: Optional[bool]=False):
    self.min_freq = min_freq
    self.special_string = special_string

    self.skipped_words = []
    self.list_string = self._counter_to_list(counter, self.min_freq)
    if special_string:
      self.list_string.extend(special_string)

    self.allow_unknown_tag = allow_unknown_tag
    self.unknown_tag = None
    if self.allow_unknown_tag:
      self.unknown_tag = '<UNK_TAG>'
      self.list_string.append(self.unknown_tag)

    if shuffle:
      random.shuffle(self.list_string)

  def is_valid_string(self, label: str) -> bool:
    return label in self.list_string

  def is_valid_int(self, index: int) -> bool:
    return index > 0 and index < len(self.list_string)

  def stoi(self, string_in: str) -> int:
    if self.allow_unknown_tag and not string_in in self.list_string:
      # unknown label 
      return self.list_string.index(self.unknown_tag)
    else:
      return self.list_string.index(string_in)

  def itos(self, int_in: int) -> str:
    return self.list_string[int_in]

  def _counter_to_list(self, counter: Counter, 
                        min_freq: Optional[int]=None) -> List[str]:
    list_string = []
    self.skipped_words = []
    for item in counter.items():
      label = item[0]
      occ = item[1]
      if min_freq:
        if occ >= min_freq:
          list_string.append(label)
        else:
          self.skipped_words.append(label)
      else:
        list_string.append(label)
    #ic(self.skipped_words);
    return list_string
  
  def __len__(self):
    return len(self.list_string)
