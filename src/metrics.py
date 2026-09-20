import numpy as np


def recall_at_k(predicted_items, relevant_items, k=50):
    predicted_set = set(predicted_items[:k])
    relevant_set = set(relevant_items)

    if not relevant_set:
        raise ValueError("Empty relevant items")
    return len(relevant_set & predicted_set) / len(relevant_set)


def mean_recall_at_k(predictions, ground_truth, k=50):
    scores = [
        recall_at_k(predicted_items, relevant_items, k)
        for predicted_items, relevant_items in zip(predictions, ground_truth)
    ]
    return float(np.mean(scores))


def evaluate_position_matrix(position_matrix, item_ids, relevant_items_per_query, k=50):
    #Считает Recall@k для матрицы позиций объявлений в корпусе.
    scores = np.empty(len(relevant_items_per_query), dtype=np.float32)

    for query_position, relevant_items in enumerate(relevant_items_per_query):
        positions = np.asarray(position_matrix[query_position, :k])
        positions = positions[positions >= 0]
        predicted_items = item_ids[positions]
        scores[query_position] = recall_at_k(predicted_items, relevant_items, k)

    return scores
