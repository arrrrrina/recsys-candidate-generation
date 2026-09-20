def build_query_text(data):
    # бъединяет текст запроса и выбранные пользователем фильтры.
    query = data["search_query"].fillna("").astype(str)
    params = data["search_infm_params_text"].fillna("").astype(str)
    return (query + " " + params).str.strip()


def build_lexical_item_text(data):
    # Короткий текст объявления для лексического TF-IDF-поиска.
    title = data["item_title_raw"].fillna("").astype(str)
    params = data["item_infm_params_text"].fillna("").astype(str)
    return (title + " " + params).str.strip()


def build_semantic_item_text(data, description_limit=500):
    # Текст объявления для E5 с ограниченным описаниtv.
    title = data["item_title_raw"].fillna("").astype(str)
    params = data["item_infm_params_text"].fillna("").astype(str)
    description = data["item_description_raw"].fillna("").astype(str).str[:description_limit]
    return (title + " " + params + " " + description).str.strip()


def build_item_text(data):
    return build_semantic_item_text(data, description_limit=None)
