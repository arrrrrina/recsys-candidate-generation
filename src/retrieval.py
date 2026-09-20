import gc
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import BallTree, NearestNeighbors


def load_semantic_model(model_name="intfloat/multilingual-e5-small", max_seq_length=256):
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    return model


def encode_items(model, item_texts, batch_size=32):
    documents = ["passage: " + text for text in item_texts]
    return model.encode(
        documents,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def encode_queries(model, query_texts, batch_size=32):
    queries = ["query: " + text for text in query_texts]
    return model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def load_or_encode_embeddings(
    item_texts,
    query_texts,
    item_path,
    query_path,
    model_name="intfloat/multilingual-e5-small",
    item_batch_size=64,
    query_batch_size=128,
    max_seq_length=256,
):
    # Загружает эмбеддинги из кэша или считает только отсутствующие массивы.
    item_path = Path(item_path)
    query_path = Path(query_path)
    model = None

    if not item_path.exists() or not query_path.exists():
        model = load_semantic_model(model_name, max_seq_length=max_seq_length)

    if not item_path.exists():
        np.save(item_path, encode_items(model, item_texts, batch_size=item_batch_size))
    if not query_path.exists():
        np.save(query_path, encode_queries(model, query_texts, batch_size=query_batch_size))

    del model
    gc.collect()
    return np.load(item_path, mmap_mode="r"), np.load(query_path, mmap_mode="r")


def load_or_build_tfidf(
    item_texts, query_texts, vectorizer_path, item_matrix_path, query_matrix_path
):
    # Создаёт или загружает TF-IDF-модель и обе разреженные матрицы.
    vectorizer_path = Path(vectorizer_path)
    item_matrix_path = Path(item_matrix_path)
    query_matrix_path = Path(query_matrix_path)

    if vectorizer_path.exists() and item_matrix_path.exists():
        vectorizer = joblib.load(vectorizer_path)
        item_matrix = load_npz(item_matrix_path).tocsr()
    else:
        vectorizer = TfidfVectorizer(lowercase=True, dtype=np.float32)
        item_matrix = vectorizer.fit_transform(item_texts)
        joblib.dump(vectorizer, vectorizer_path)
        save_npz(item_matrix_path, item_matrix)

    if query_matrix_path.exists():
        query_matrix = load_npz(query_matrix_path).tocsr()
    else:
        query_matrix = vectorizer.transform(query_texts)
        save_npz(query_matrix_path, query_matrix)

    return vectorizer, item_matrix, query_matrix


def _group_positions(location_ids, category_ids):
    return pd.DataFrame(
        {"location": location_ids, "category": category_ids}
    ).groupby(["location", "category"], sort=False).indices


def load_or_build_global_semantic(
    item_embeddings,
    query_embeddings,
    indices_path,
    scores_path,
    index_path,
    top_k=100,
):
    # Строит глобальный HNSW-поиск или загружает готовый top-k.
    indices_path = Path(indices_path)
    scores_path = Path(scores_path)
    index_path = Path(index_path)

    if indices_path.exists() and scores_path.exists():
        return np.load(indices_path, mmap_mode="r"), np.load(scores_path, mmap_mode="r")

    faiss.omp_set_num_threads(4)
    if index_path.exists():
        search_index = faiss.read_index(str(index_path))
    else:
        search_index = faiss.IndexHNSWFlat(
            item_embeddings.shape[1], 16, faiss.METRIC_INNER_PRODUCT
        )
        search_index.hnsw.efConstruction = 80
        for start in range(0, len(item_embeddings), 5_000):
            batch = np.ascontiguousarray(item_embeddings[start:start + 5_000], dtype=np.float32)
            search_index.add(batch)
        faiss.write_index(search_index, str(index_path))

    search_index.hnsw.efSearch = 128
    scores, indices = search_index.search(
        np.ascontiguousarray(query_embeddings, dtype=np.float32), top_k
    )
    np.save(indices_path, indices.astype(np.int32))
    np.save(scores_path, scores.astype(np.float32))
    del search_index
    gc.collect()
    return np.load(indices_path, mmap_mode="r"), np.load(scores_path, mmap_mode="r")


def load_or_build_local_semantic(
    item_embeddings,
    query_embeddings,
    item_location_ids,
    item_category_ids,
    query_location_ids,
    query_category_ids,
    indices_path,
    scores_path,
    top_k=40,
):
    # Ищет точных E5-соседей внутри одинаковых location и category.
    indices_path = Path(indices_path)
    scores_path = Path(scores_path)
    if indices_path.exists() and scores_path.exists():
        return np.load(indices_path, mmap_mode="r"), np.load(scores_path, mmap_mode="r")

    working_indices = indices_path.with_suffix(".working.npy")
    working_scores = scores_path.with_suffix(".working.npy")
    indices = np.lib.format.open_memmap(
        working_indices, mode="w+", dtype=np.int32, shape=(len(query_embeddings), top_k)
    )
    scores = np.lib.format.open_memmap(
        working_scores, mode="w+", dtype=np.float32, shape=(len(query_embeddings), top_k)
    )
    indices[:] = -1
    scores[:] = -np.inf

    item_groups = _group_positions(item_location_ids, item_category_ids)
    query_groups = _group_positions(query_location_ids, query_category_ids)
    faiss.omp_set_num_threads(4)

    for group_number, (group_key, query_positions) in enumerate(query_groups.items()):
        item_positions = item_groups.get(group_key)
        if item_positions is None:
            continue

        item_positions = np.asarray(item_positions, dtype=np.int64)
        query_positions = np.asarray(query_positions, dtype=np.int64)
        neighbors_count = min(top_k, len(item_positions))
        search_index = faiss.IndexFlatIP(item_embeddings.shape[1])

        for start in range(0, len(item_positions), 5_000):
            positions = item_positions[start:start + 5_000]
            search_index.add(
                np.ascontiguousarray(item_embeddings[positions], dtype=np.float32)
            )

        group_scores, relative_indices = search_index.search(
            np.ascontiguousarray(query_embeddings[query_positions], dtype=np.float32),
            neighbors_count,
        )
        indices[query_positions, :neighbors_count] = item_positions[relative_indices]
        scores[query_positions, :neighbors_count] = group_scores
        del search_index

        if group_number % 100 == 0:
            indices.flush()
            scores.flush()

    indices.flush()
    scores.flush()
    del indices, scores
    working_indices.replace(indices_path)
    working_scores.replace(scores_path)
    return np.load(indices_path, mmap_mode="r"), np.load(scores_path, mmap_mode="r")


def load_or_build_local_tfidf(
    item_matrix,
    query_matrix,
    item_location_ids,
    item_category_ids,
    query_location_ids,
    query_category_ids,
    indices_path,
    scores_path,
    top_k=40,
):
    # Ищет TF-IDF-соседей внутри одинаковых location и category.
    indices_path = Path(indices_path)
    scores_path = Path(scores_path)
    if indices_path.exists() and scores_path.exists():
        return np.load(indices_path, mmap_mode="r"), np.load(scores_path, mmap_mode="r")

    indices = np.lib.format.open_memmap(
        indices_path, mode="w+", dtype=np.int32, shape=(query_matrix.shape[0], top_k)
    )
    scores = np.lib.format.open_memmap(
        scores_path, mode="w+", dtype=np.float32, shape=(query_matrix.shape[0], top_k)
    )
    indices[:] = -1
    scores[:] = -np.inf

    item_groups = _group_positions(item_location_ids, item_category_ids)
    query_groups = _group_positions(query_location_ids, query_category_ids)

    for group_number, (group_key, query_positions) in enumerate(query_groups.items()):
        item_positions = item_groups.get(group_key)
        if item_positions is None:
            continue

        item_positions = np.asarray(item_positions)
        query_positions = np.asarray(query_positions)
        neighbors_count = min(top_k, len(item_positions))
        search_index = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=4)
        search_index.fit(item_matrix[item_positions])
        distances, relative_indices = search_index.kneighbors(
            query_matrix[query_positions], n_neighbors=neighbors_count
        )
        indices[query_positions, :neighbors_count] = item_positions[relative_indices]
        scores[query_positions, :neighbors_count] = 1 - distances

        if group_number % 100 == 0:
            indices.flush()
            scores.flush()

    indices.flush()
    scores.flush()
    return indices, scores


def load_or_build_nearby_semantic(
    item_embeddings,
    query_embeddings,
    item_location_ids,
    item_category_ids,
    query_location_ids,
    query_category_ids,
    item_latitudes,
    item_longitudes,
    query_latitudes,
    query_longitudes,
    indices_path,
    top_k=20,
    radius_km=100.0,
):
   # Ищет E5-соседей той же категории в соседних локациях
    indices_path = Path(indices_path)
    if indices_path.exists():
        return np.load(indices_path, mmap_mode="r")

    working_path = indices_path.with_suffix(".working.npy")
    indices = np.lib.format.open_memmap(
        working_path, mode="w+", dtype=np.int32, shape=(len(query_embeddings), top_k)
    )
    indices[:] = -1

    valid_item_positions = np.flatnonzero(
        np.isfinite(item_latitudes) & np.isfinite(item_longitudes)
    )
    item_coordinates = np.radians(
        np.column_stack(
            [item_latitudes[valid_item_positions], item_longitudes[valid_item_positions]]
        )
    )
    geo_index = BallTree(item_coordinates, metric="haversine", leaf_size=40)
    query_groups = _group_positions(query_location_ids, query_category_ids)
    radius_radians = radius_km / 6371.0
    faiss.omp_set_num_threads(4)

    for group_number, ((location_id, category_id), query_positions) in enumerate(
        query_groups.items()
    ):
        query_positions = np.asarray(query_positions, dtype=np.int64)
        finite = np.isfinite(query_latitudes[query_positions]) & np.isfinite(
            query_longitudes[query_positions]
        )
        if not finite.any():
            continue

        finite_positions = query_positions[finite]
        center = np.radians(
            [[
                np.nanmedian(query_latitudes[finite_positions]),
                np.nanmedian(query_longitudes[finite_positions]),
            ]]
        )
        relative_positions = geo_index.query_radius(center, r=radius_radians)[0]
        candidate_positions = valid_item_positions[relative_positions]
        candidate_positions = candidate_positions[
            (item_category_ids[candidate_positions] == category_id)
            & (item_location_ids[candidate_positions] != location_id)
        ]
        if len(candidate_positions) == 0:
            continue

        neighbors_count = min(top_k, len(candidate_positions))
        search_index = faiss.IndexFlatIP(item_embeddings.shape[1])
        for start in range(0, len(candidate_positions), 5_000):
            positions = candidate_positions[start:start + 5_000]
            search_index.add(
                np.ascontiguousarray(item_embeddings[positions], dtype=np.float32)
            )
        _, relative_indices = search_index.search(
            np.ascontiguousarray(query_embeddings[query_positions], dtype=np.float32),
            neighbors_count,
        )
        indices[query_positions, :neighbors_count] = candidate_positions[relative_indices]
        del search_index

        if group_number % 100 == 0:
            indices.flush()
            gc.collect()

    indices.flush()
    del indices
    working_path.replace(indices_path)
    return np.load(indices_path, mmap_mode="r")


def build_tfidf_index(item_texts):
    vectorizer = TfidfVectorizer(lowercase=True, dtype=np.float32)
    item_matrix = vectorizer.fit_transform(item_texts)
    search_index = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    search_index.fit(item_matrix)
    return vectorizer, item_matrix, search_index


def retrieve_global_tfidf(query_texts, vectorizer, search_index, top_k=10):
    query_matrix = vectorizer.transform(query_texts)
    distances, indices = search_index.kneighbors(query_matrix, n_neighbors=top_k)
    return indices, 1 - distances, query_matrix


def build_semantic_index(item_embeddings):
    search_index = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    search_index.fit(item_embeddings)
    return search_index


def retrieve_semantic(query_embeddings, search_index, top_k=10):
    distances, indices = search_index.kneighbors(query_embeddings, n_neighbors=top_k)
    return indices, 1 - distances
