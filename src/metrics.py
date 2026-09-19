def recall_at_k(predicted_items, relevant_items, k=50):
    predicted_set = set(predicted_items[:k])
    relevant_set = set(relevant_items)

    if len(relevant_set) == 0:
        raise ValueError("Empty relevant items")
    
    found_items = relevant_set & predicted_set

    return len(found_items) / len(relevant_set)

def mean_recall_at_k(predictions, ground_truth, k=50):
    scores = []

    for predicted_items, relevant_items in zip(predictions, ground_truth):
        score = recall_at_k(predicted_items, relevant_items, k)
        scores.append(score)

    return sum(scores) / len(scores)
