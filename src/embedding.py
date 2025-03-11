import re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = PorterStemmer()

    def pre_processing(self, doc):
        pattern = re.compile(r'\d+|http\S+|<.*?>', re.IGNORECASE)
        return pattern.sub('', doc).lower()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [' '.join(self.stemmer.stem(word) for word in word_tokenize(self.pre_processing(doc))) for doc in X]

class Lazy(BaseEstimator, TransformerMixin):

    def __init__(self, vectorizer, ngram_range, max_df, min_df, max_features):
        self.vectorizer = vectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features)
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features

    def fit(self, raw_documents, y=None):
        self.vectorizer.fit(raw_documents, y)
        return self

    def transform(self, raw_documents):
        return self.vectorizer.transform(raw_documents).toarray()

    def fit_transform(self, raw_documents, y=None):
        return self.vectorizer.fit_transform(raw_documents, y).toarray()

class BoW(Lazy):
    def __init__(self, ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=9000):
        super().__init__(CountVectorizer, ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features)

class TFIDF(Lazy):
    def __init__(self, ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=9000):
        super().__init__(TfidfVectorizer, ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features)

class Embedding(BaseEstimator, TransformerMixin):
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.model.encode(text) for text in X])

class RoBERTa(Embedding):
    def __init__(self, model_name='all-distilroberta-v1'):
        super().__init__(model_name=model_name)

class USE(Embedding):
    def __init__(self, model_name='distiluse-base-multilingual-cased-v1'):
        super().__init__(model_name=model_name)
