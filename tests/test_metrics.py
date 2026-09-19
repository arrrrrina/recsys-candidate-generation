from src.metrics import recall_at_k


def test_recall_finds_half_of_relevant_items():
    predictions = ["a", "b"]
    relevant_items = ["b", "c"]

    result = recall_at_k(
        predictions,
        relevant_items,
        k=2,
    )

    assert result == 0.5


def test_recall_is_one_when_all_items_found():
    predictions = ["a", "b", "c"]
    relevant_items = ["a", "c"]

    result = recall_at_k(
        predictions,
        relevant_items,
        k=3,
    )

    assert result == 1.0


def test_recall_respects_k():
    predictions = ["a", "b", "c"]
    relevant_items = ["c"]

    result = recall_at_k(
        predictions,
        relevant_items,
        k=2,
    )

    assert result == 0.0