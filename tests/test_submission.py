import pandas as pd
import pytest

from src.submission import build_answer, validate_answer


def test_valid_answer_passes_checks():
    query_ids = ["0123456789abcdef", "fedcba9876543210"]
    item_ids = ["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"]
    answer = build_answer(query_ids, [[item_ids[0]], [item_ids[1]]])

    predictions = validate_answer(answer, query_ids, item_ids)

    assert predictions.tolist() == [[item_ids[0]], [item_ids[1]]]


def test_unknown_item_is_rejected():
    answer = pd.DataFrame(
        {"query_id": ["0123456789abcdef"], "answer": ["cccccccccccccccc"]}
    )

    with pytest.raises(ValueError, match="отсутствующие"):
        validate_answer(answer, ["0123456789abcdef"], ["aaaaaaaaaaaaaaaa"])
