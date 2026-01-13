from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
import numpy as np


def fit_vectorizer(corpus: List[str], max_features: int = 5000) -> Tuple[TfidfVectorizer, np.ndarray]:
    vect = TfidfVectorizer(max_features=max_features)
    X = vect.fit_transform(corpus)
    return vect, X


def transform_documents(vect: TfidfVectorizer, docs: List[str]):
    return vect.transform(docs)
