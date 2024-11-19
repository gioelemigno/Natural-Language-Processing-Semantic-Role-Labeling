from typing import *
import nltk

class POS_Tagger():
  """
  Wrapper of nltk POS Tagger
  # Attributes
  
    `pos_tags_set_name`: `str`, optional, (default=None)
        Name of the POS TAGS set to use
  """

  def __init__(self, pos_tags_set_name=None):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')
    nltk.download('tagsets')

    self.pos_tags_set_name = pos_tags_set_name

    if self.pos_tags_set_name == 'universal':
      # POS tags
      # Universal POS Tags
      # https://www.nltk.org/_modules/nltk/tag/mapping.html
      '''
      self.pos_tag_list = [
                            "VERB",
                            "NOUN",
                            "PRON",
                            "ADJ",
                            "ADV",
                            "ADP",
                            "CONJ",
                            "DET",
                            "NUM",
                            "PRT",
                            "X",
                            "."]
      '''
      pass
    elif self.pos_tags_set_name is None:
      pass
    else:
      print("error unknown pos_tags_set_name")
      return 

  def pos_tagging(self, 
                  sentence: List[str]):      
    """
    Used to perform POS Tagging
    # Parameters
        `sentence`: `List[str]`, required
          Input sentence splitted in words

    # Returns 
        `List[str]`
          List of POS Tags 
    """  

    pos_nltk = nltk.tag.pos_tag(sentence, tagset=self.pos_tags_set_name)
    pos = [pos[1] for pos in pos_nltk]
    return pos
