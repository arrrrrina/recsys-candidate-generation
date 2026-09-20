from pathlib import Path

import pandas as pd


def positions_to_predictions(position_matrix, item_ids, top_k=50):
    predictions = []

    for position_row in position_matrix:
        items = []
        seen = set()
        for position in position_row:
            if position < 0:
                continue
            item_id = str(item_ids[position])
            if item_id not in seen:
                items.append(item_id)
                seen.add(item_id)
            if len(items) == top_k:
                break
        predictions.append(items)

    return predictions


def build_answer(query_ids, predictions):
    return pd.DataFrame(
        {
            "query_id": [str(query_id) for query_id in query_ids],
            "answer": [" ".join(items) for items in predictions],
        }
    )


def validate_answer(answer, expected_query_ids, available_item_ids, top_k=50):
    # проверка файла на корректность
    if answer.columns.tolist() != ["query_id", "answer"]:
        raise ValueError("answer.csv должен содержать только query_id и answer")
    if len(answer) != len(expected_query_ids) or not answer["query_id"].is_unique:
        raise ValueError("Количество или уникальность query_id нарушены")
    if set(answer["query_id"]) != set(map(str, expected_query_ids)):
        raise ValueError("Набор query_id не совпадает с benchmark")
    if not answer["query_id"].str.len().eq(16).all():
        raise ValueError("query_id должен состоять ровно из 16 символов")

    predictions = answer["answer"].map(lambda value: value.split() if value else [])
    if not predictions.map(len).le(top_k).all():
        raise ValueError(f"В одном из ответов больше {top_k} item_id")
    if not predictions.map(lambda items: len(items) == len(set(items))).all():
        raise ValueError("Внутри ответа найдены повторяющиеся item_id")

    predicted_items = [item for items in predictions for item in items]
    available_items = set(map(str, available_item_ids))
    if not set(predicted_items).issubset(available_items):
        raise ValueError("В ответе есть item_id, отсутствующие в benchmark_items")
    if not all(len(item_id) == 16 for item_id in predicted_items):
        raise ValueError("item_id должен состоять ровно из 16 символов")
    if not all(set(item_id) <= set("0123456789abcdef") for item_id in predicted_items):
        raise ValueError("item_id содержит недопустимые символы")

    return predictions


def save_answer(answer, path):
    path = Path(path)
    answer.to_csv(path, index=False, encoding="utf-8")
    return path
