from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.portstem = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, content):
        content = re.sub('[^a-zA-Z]', ' ', str(content))
        content = content.lower()
        words = content.split()
        words = [self.portstem.stem(word) for word in words if word not in self.stop_words]
        return ' '.join(words)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_text(text) for text in X]
