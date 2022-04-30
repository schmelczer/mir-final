from typing import List, Tuple

import numpy as np
from gensim import corpora, matutils, models
from nltk.stem import WordNetLemmatizer

from captioned_image import CaptionedImage

lemmatizer = WordNetLemmatizer()


def train_tfidf(
    data: List[CaptionedImage],
) -> Tuple[corpora.Dictionary, models.TfidfModel]:
    texts = [_preprocess_text(c) for d in data for c in d.captions]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus, smartirs='txc')
    return dictionary, tfidf


def get_tfidf_vector(
    model: Tuple[corpora.Dictionary, models.TfidfModel], text: str
) -> np.ndarray:
    dictionary, tfidf = model
    tokens = _preprocess_text(text)
    bow = dictionary.doc2bow(tokens)
    transformed_bow = tfidf[bow]
    return matutils.sparse2full(transformed_bow, len(dictionary))


def _preprocess_text(text: str) -> List[str]:
    return [lemmatizer.lemmatize(word.lower().strip()) for word in text.split()]

