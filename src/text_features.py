import pandas as pd

# Подготавливает запрос и его параметры для энкодера, Nan заменяет пустой строкой
def build_query_text(data):
    query = data["search_query"].fillna("")
    params = data["search_infm_params_text"].fillna("")

    return query + " " + params

# Подготавливает запрос и его параметры для энкодера, Nan заменяет пустой строкой
def build_item_text(data):
    title = data["item_title_raw"].fillna("")
    params = data["item_infm_params_text"].fillna("")
    description = data["item_description_raw"].fillna("")

    return title + " " + params + " " + description
# Для TF-IDF
def build_lexical_item_text(data):
    title = data["item_title_raw"].fillna("")
    params = data["item_infm_params_text"].fillna("")

    return title + " " + params
# Для Е5
def build_semantic_item_text(data):
    title = data["item_title_raw"].fillna("")
    params = data["item_infm_params_text"].fillna("")

    description = (
        data["item_description_raw"]
        .fillna("")
        .str[:500]
    )

    return title + " " + params + " " + description