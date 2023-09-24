
import numpy as np
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

class NLPTransformer(BaseEstimator, TransformerMixin):
  """
  Transform any text into normalized text
  1. Remove number, punctuation and lower case text
  2. Lemmatization or stem
  3. Remove stop words
  Args:
      - `language`: language of text
      - `stopwords`: the words that will be remove from text
      - `stem`: whether to use `stemmer` or `lemmatization`
          from NLTK default `stemmer`. if `None` no stemmer
          or lemmatization will be applied
  """
  def __init__(self, language="english", stopwords=None, stem=True):
    super().__init__()

    # We need to download stopwords first
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)

    # Lemmatizer
    self.nltk_lemmatizer = WordNetLemmatizer()

    # language
    self.language = language

    # using stem
    self.stem = stem

    # custom stopwords
    self.stopwords = stopwords

    # List of stopwords from
    # nltk by language
    self.stopwords_list = set(nltk.corpus.stopwords.words(language))

    # Add custom words
    if self.stopwords != None:
      self.stopwords_list.update(self.stopwords)

    # regex pattern
    self.pattern = r"('(?:\w+))|\\r\\n|\\n|\\r|[^a-zA-Z]"

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    new_X = np.copy(X)
    if new_X.ndim != 1:
      warnings.warn("Input X is not 1 dimensional array try to reshape array")
      new_X = new_X.reshape(-1)

    new_X = self.remove_num_punc(new_X)

    if self.stem is not None:
      if self.stem:
        new_X = self.stemming(new_X, self.language)
      else:
        new_X = self.lemmatize_words(new_X, self.language)

    new_X = self.remove_stop_words(new_X, self.stopwords_list)

    return new_X

  # remove numbers and punctuations
  def remove_num_punc(self, corpus):
    '''
    Remove all numbers and punctuations from given words
    '''
    for i, text in enumerate(corpus):
      # remove all numbers and punctuations
      clean_words = re.sub(self.pattern,' ',str(text), flags=re.S)

      # turn words into lower case and split them into array
      words_arr = clean_words.lower().split()

      # join words with white space between each words
      join_words = ' '.join(words_arr)
      join_words = str.lower(join_words)

      corpus[i] = join_words

    return corpus

  # remove stop words
  def remove_stop_words(self, corpus, stopwords):
    '''
    Remove stop words from words
    '''
    for i, text in enumerate(corpus):
      # Split string into words
      word_arr = text.split()

      # Ignore words in stop words list
      filter_words = [word for word in word_arr
                        if word not in stopwords]
      join_words = ' '.join(filter_words)

      corpus[i] = join_words

    return corpus

  def get_wordnet_pos(self, treebank_tag):
    '''
    Treebank tag
    https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    '''
    if treebank_tag.startswith('J'):
          return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    elif treebank_tag.startswith('VBZ'):
        return wordnet.ADJ_SAT
    else:
        return ''

  def tokenize_words(self, text, lang):
    return word_tokenize(text, lang)

  def lemmatize_words(self, corpus, lang):
    '''
    Lemmatize words

    Return a lemmatized words
    '''
    for i, text in enumerate(corpus):
      # tokenize words
      tokenized_words = self.tokenize_words(text, lang)

      # lemmatize words
      lemmatized_words = []

      # for word in tokenized_words:
      #   lemmatized_words.append(lemmatizer.lemmatize(word))

      # Pos
      for word, tag in nltk.tag.pos_tag(tokenized_words):
        pos = self.get_wordnet_pos(tag)
        if pos == '':
          lemmatized_words.append(word)
        else:
          lemmatized_words.append(self.nltk_lemmatizer.lemmatize(word, pos=pos))

      # join words together
      final_sentence = ' '.join(lemmatized_words)
      corpus[i] = final_sentence

    return corpus

  def stemming(self, corpus, lang):
    '''
    Stem words

    Return a text with words that is stemmed
    '''
    for i, text in enumerate(corpus):
      # tokenize words
      tokenized_words = self.tokenize_words(text, lang)

      #stem words
      stemmer = SnowballStemmer(lang)
      stemmed_words = []

      for word in tokenized_words:
        stemmed_words.append(stemmer.stem(word))

      # join words together
      joined_words =  ' '.join(stemmed_words)

      corpus[i] = joined_words

    return corpus
