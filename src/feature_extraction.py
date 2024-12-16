from preprocess import *
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from utils import read_test_cases as rtc
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch

def extract_tfidf_features(data):
    """Extract TF-IDF features."""
    vectorizer = TfidfVectorizer(lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(data)
    return tfidf_matrix, vectorizer.get_feature_names_out()

def extract_embeddings_word2vec(data):
    """Extract embeddings Word2Vec"""
    model = Word2Vec(data, min_count=1, sg=0, hs=1)
    word_vectors = model.wv.vectors
    normalized_word_vectors = normalize(word_vectors)
    model.wv.vectors = normalized_word_vectors
    return model


def contextual_embeddings(data):
    """Extract contextual embeddings using RoBERTa."""
    # Load pre-trained model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    # Tokenize the input data
    inputs = tokenizer(data, return_tensors='pt', padding=True, truncation=True)
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # The last hidden state is the contextual embeddings
    embeddings = outputs.last_hidden_state
    return embeddings


def hybrid_features(data):
    """Extract hybrid features by combining two different features"""


if __name__ == '__main__':
    path = './try.json'
    data = preprocess_text(path)
    data_as_strings = [' '.join(tokens) for tokens in data]

    tfidf_matrix, feature_names = extract_tfidf_features(data_as_strings)
    print("TF-IDF Matrix:\n", tfidf_matrix.toarray())
    print("Feature Names:\n", feature_names)

    word2vec_model = extract_embeddings_word2vec(data)
    print("Word2Vec Model:\n", word2vec_model)

    text = rtc(path)
    cleaned = clean_text(text)
    embeddings = contextual_embeddings(cleaned)
    print("Contextual Embeddings:\n", embeddings)