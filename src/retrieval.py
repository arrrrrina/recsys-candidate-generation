from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sentence_transformers import SentenceTransformer

def build_tfidf_index(item_texts):
    vectorizer = TfidfVectorizer(lowercase=True, dtype=np.float32)
    item_matrix = vectorizer.fit_transform(item_texts)

    search_index = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    search_index.fit(item_matrix)

    return vectorizer, item_matrix, search_index

def retrieve_global_tfidf(query_texts, vectorizer, search_index, top_k=10,):
    query_matrix = vectorizer.transform(query_texts)

    distances, indices = search_index.kneighbors(query_matrix, n_neighbors=top_k)
    similarities = 1 - distances

    return indices, similarities, query_matrix

# Для семантического канала
def load_semantic_model():
    return SentenceTransformer("intfloat/multilingual-e5-small")


def encode_items(model, item_texts, batch_size=32):
    documents = ["passage: " + text for text in item_texts]

    return model.encode(
        documents,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def encode_queries(model, query_texts, batch_size=32,):
    queries = ["query: " + text for text in query_texts]

    return model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def build_semantic_index(item_embeddings):
    search_index = NearestNeighbors( metric="cosine", algorithm="brute", n_jobs=-1)

    search_index.fit(item_embeddings)

    return search_index


def retrieve_semantic(query_embeddings, search_index, top_k=10):
    distances, indices = search_index.kneighbors(query_embeddings, n_neighbors=top_k)

    similarities = 1 - distances

    return indices, similarities