from sklearn.feature_extraction.text import  TfidfVectorizer
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
import pandas as pd

from nltk.stem.porter import PorterStemmer

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


def create_tf_idf_model(list_text):
  # create the transform
  tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=1000, use_idf=True,
                              norm='l2', stop_words=stopwords.words("spanish"), tokenizer=tokenize)

  # tokenize and build vocab
  tfidf.fit(list_text)
  return tfidf


def convert_text_to_features(df, tfidf, text_column_name='text'):
  vector = tfidf.transform(df[text_column_name]).todense()
  new_cols = tfidf.get_feature_names_out()
  df = df.drop(text_column_name, axis=1)
  return df.join(pd.DataFrame(vector, columns=new_cols))
