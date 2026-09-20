import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "local_e5_reciprocal_rank",
    "local_tfidf_reciprocal_rank",
    "nearby_e5_reciprocal_rank",
    "global_e5_reciprocal_rank",
    "in_local_e5",
    "in_local_tfidf",
    "in_nearby_e5",
    "in_global_e5",
    "local_e5_score",
    "local_tfidf_score",
    "global_e5_score",
    "channels_count",
    "same_location",
    "same_category",
]

CHANNEL_ORDER = ["local_e5", "local_tfidf", "nearby_e5", "global_e5"]
SCORE_COLUMNS = {"local_e5": 8, "local_tfidf": 9, "global_e5": 10}


def build_candidate_features(
    query_position,
    channels,
    item_location_ids,
    query_location_ids,
    item_category_ids,
    query_category_ids,
):
    # Строит признаки для объединённого пула одного запроса.
    features_by_item = {}

    for channel_number, channel_name in enumerate(CHANNEL_ORDER):
        indices = channels[channel_name]["indices"]
        scores = channels[channel_name].get("scores")

        for rank, item_position in enumerate(indices[query_position], start=1):
            if item_position < 0:
                continue

            item_position = int(item_position)
            if item_position not in features_by_item:
                features_by_item[item_position] = np.zeros(
                    len(FEATURE_NAMES), dtype=np.float32
                )

            features = features_by_item[item_position]
            features[channel_number] = 1.0 / rank
            features[4 + channel_number] = 1.0

            score_column = SCORE_COLUMNS.get(channel_name)
            if scores is not None and score_column is not None:
                score = scores[query_position, rank - 1]
                if np.isfinite(score):
                    features[score_column] = score

    item_positions = np.fromiter(features_by_item.keys(), dtype=np.int32)
    feature_matrix = np.vstack(list(features_by_item.values()))
    feature_matrix[:, 11] = feature_matrix[:, 4:8].sum(axis=1)
    feature_matrix[:, 12] = (
        item_location_ids[item_positions] == query_location_ids[query_position]
    )
    feature_matrix[:, 13] = (
        item_category_ids[item_positions] == query_category_ids[query_position]
    )
    return item_positions, feature_matrix


def build_training_matrix(
    query_positions,
    relevant_items_per_query,
    item_ids,
    channels,
    item_location_ids,
    query_location_ids,
    item_category_ids,
    query_category_ids,
    negatives_per_query=20,
    random_state=42,
):
    # Формирует positive/negative пары.
    rng = np.random.default_rng(random_state)
    feature_parts = []
    label_parts = []
    queries_with_positive = 0

    for query_position in query_positions:
        item_positions, features = build_candidate_features(
            query_position,
            channels,
            item_location_ids,
            query_location_ids,
            item_category_ids,
            query_category_ids,
        )
        relevant_items = set(relevant_items_per_query[query_position])
        labels = np.fromiter(
            (item_ids[position] in relevant_items for position in item_positions),
            dtype=np.int8,
        )
        positive_rows = np.flatnonzero(labels == 1)
        negative_rows = np.flatnonzero(labels == 0)

        if len(positive_rows) == 0:
            continue

        queries_with_positive += 1
        sampled_negatives = rng.choice(
            negative_rows,
            size=min(negatives_per_query, len(negative_rows)),
            replace=False,
        )
        selected_rows = np.concatenate([positive_rows, sampled_negatives])
        feature_parts.append(features[selected_rows])
        label_parts.append(labels[selected_rows])

    return (
        np.vstack(feature_parts).astype(np.float32),
        np.concatenate(label_parts),
        queries_with_positive,
    )


def train_linear_blender(features, labels, random_state=42):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            random_state=random_state,
        ),
    )
    return model.fit(features, labels)


def rank_with_model(
    model,
    number_of_queries,
    channels,
    item_location_ids,
    query_location_ids,
    item_category_ids,
    query_category_ids,
    query_positions=None,
    top_k=50,
    batch_size=500,
):
    # Ранжирует объединённый пул и возвращает позиции top-k.
    if query_positions is None:
        query_positions = np.arange(number_of_queries)
    query_positions = np.asarray(query_positions)
    result = np.full((number_of_queries, top_k), -1, dtype=np.int32)

    for batch_start in range(0, len(query_positions), batch_size):
        batch_queries = query_positions[batch_start:batch_start + batch_size]
        feature_parts = []
        candidate_parts = []

        for query_position in batch_queries:
            candidates, features = build_candidate_features(
                query_position,
                channels,
                item_location_ids,
                query_location_ids,
                item_category_ids,
                query_category_ids,
            )
            candidate_parts.append(candidates)
            feature_parts.append(features)

        probabilities = model.predict_proba(np.vstack(feature_parts))[:, 1]
        offset = 0

        for query_position, candidates in zip(batch_queries, candidate_parts):
            query_probabilities = probabilities[offset:offset + len(candidates)]
            offset += len(candidates)
            ranked = candidates[np.argsort(-query_probabilities, kind="stable")]

            same_category = ranked[
                item_category_ids[ranked] == query_category_ids[query_position]
            ]
            other_categories = ranked[
                item_category_ids[ranked] != query_category_ids[query_position]
            ]
            selected = np.concatenate([same_category, other_categories])[:top_k]
            result[query_position, :len(selected)] = selected

    return result


def build_quota_fusion(
    channels,
    item_category_ids,
    query_category_ids,
    quotas=(30, 5, 10),
    top_k=50,
):
    # Простой baseline: local E5, local TF-IDF, nearby E5 и global fallback.
    number_of_queries = len(query_category_ids)
    result = np.full((number_of_queries, top_k), -1, dtype=np.int32)
    local_names = ["local_e5", "local_tfidf", "nearby_e5"]

    for query_position in range(number_of_queries):
        selected = []
        selected_set = set()

        for channel_name, quota in zip(local_names, quotas):
            added = 0
            for position in channels[channel_name]["indices"][query_position]:
                if position < 0 or added >= quota or len(selected) >= top_k:
                    break
                position = int(position)
                if position not in selected_set:
                    selected.append(position)
                    selected_set.add(position)
                    added += 1

        global_positions = channels["global_e5"]["indices"][query_position]
        for require_same_category in [True, False]:
            for position in global_positions:
                if len(selected) >= top_k:
                    break
                position = int(position)
                same_category = (
                    item_category_ids[position] == query_category_ids[query_position]
                )
                if same_category == require_same_category and position not in selected_set:
                    selected.append(position)
                    selected_set.add(position)

        result[query_position, :len(selected)] = selected

    return result


def build_weighted_rrf(
    channels,
    item_category_ids,
    query_category_ids,
    weights=(3.0, 0.5, 1.5, 0.5),
    rrf_k=30,
    top_k=50,
):
    # Объединяет каналы по взвешенной Reciprocal Rank Fusion.
    number_of_queries = len(query_category_ids)
    result = np.full((number_of_queries, top_k), -1, dtype=np.int32)

    for query_position in range(number_of_queries):
        scores = {}
        for channel_name, weight in zip(CHANNEL_ORDER, weights):
            for rank, position in enumerate(
                channels[channel_name]["indices"][query_position], start=1
            ):
                if position >= 0:
                    position = int(position)
                    scores[position] = scores.get(position, 0.0) + weight / (rrf_k + rank)

        ranked = np.asarray(
            sorted(scores, key=scores.get, reverse=True), dtype=np.int32
        )
        same_category = ranked[
            item_category_ids[ranked] == query_category_ids[query_position]
        ]
        other_categories = ranked[
            item_category_ids[ranked] != query_category_ids[query_position]
        ]
        selected = np.concatenate([same_category, other_categories])[:top_k]
        result[query_position, :len(selected)] = selected

    return result


def union_recall(channels, item_ids, relevant_items_per_query):
    # Recall объединения каналов до ограничения в 50 кандидатов.
    scores = np.empty(len(relevant_items_per_query), dtype=np.float32)
    pool_sizes = np.empty(len(relevant_items_per_query), dtype=np.int32)

    for query_position, relevant_items in enumerate(relevant_items_per_query):
        pool = {
            int(position)
            for channel_name in CHANNEL_ORDER
            for position in channels[channel_name]["indices"][query_position]
            if position >= 0
        }
        pool_sizes[query_position] = len(pool)
        found = set(item_ids[list(pool)]) & set(relevant_items)
        scores[query_position] = len(found) / len(relevant_items)

    return scores, pool_sizes
