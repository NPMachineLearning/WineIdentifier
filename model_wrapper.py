
from pickle import load
# must import
from nltk_text_transformer import NLPTransformer
import nltk
import pandas as pd

class ModelWrapper:
  def __init__(self, model_path):
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)

    with open(model_path, "rb") as file:
      loaded_pkg = load(file)
      self.data = loaded_pkg["data"]
      self.pipeline = loaded_pkg["pipeline"]

    self.output_cols = ["country",
                        "province",
                        "designation",
                        "variety",
                        "winery",
                        "region"]

  def predict(self, text, filter=None):
    target = self.pipeline.predict([text])[0]
    target_data = self.data[self.data["target"]==target]
    if filter is not None:
      target_data = self.filter(target_data, filter)
    return target_data[self.output_cols]

  def filter(self, df, filter):
    for key in filter.keys():
      df = df[df[key]==filter[key]]
    return df

  def to_json(self, df):
    if isinstance(df, (pd.DataFrame, pd.Series)):
      return df.to_json(orient="records")
    return df

  def datasets(self):
    return self.data

  def country(self):
    return self.data["country"].unique()

  def province(self):
    return self.data["province"].unique()

  def designation(self):
    return self.data["designation"].unique()

  def variety(self):
    return self.data["variety"].unique()

  def winery(self):
    return self.data["winery"].unique()

  def region(self):
    return self.data["region"].unique()
