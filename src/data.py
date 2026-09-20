import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.geo import build_location_centers


SEARCH_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

ITEM_COLUMNS = [
    "item_id",
    "item_title_raw",
    "item_rating_reviews_count",
    "item_rating",
    "item_price",
    "item_microcat_id",
    "item_longitude",
    "item_location_id",
    "item_latitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
    "item_infm_params_text",
    "item_description_raw",
    "item_category_id",
]


def split_by_search_context(train, test_size=0.2, random_state=42):
    # Делит train так, чтобы полный поисковый контекст не пересекался.
    group_ids = train.groupby(SEARCH_COLUMNS, dropna=False).ngroup()
    development_ids, validation_ids = train_test_split(
        group_ids.unique(), test_size=test_size, random_state=random_state
    )
    development = train[group_ids.isin(development_ids)].copy()
    validation = train[group_ids.isin(validation_ids)].copy()
    return development, validation


def build_candidate_items(train):
    # Оставляет одну актуальную строку признаков на каждый item_id.
    return (
        train[ITEM_COLUMNS]
        .drop_duplicates("item_id", keep="last")
        .reset_index(drop=True)
    )


def build_validation_queries(validation_rows):
    # Собирает все релевантные item_id для каждого validation-запроса.
    return (
        validation_rows
        .drop_duplicates(SEARCH_COLUMNS + ["item_id"])
        .groupby(SEARCH_COLUMNS, dropna=False)
        .agg(relevant_item_ids=("item_id", list))
        .reset_index()
    )


def prepare_coordinates(queries, items, location_reference_items, fallback_interactions):
    # Восстанавливает координаты запросов и пропуски координат объявлений.
    centers = build_location_centers(location_reference_items)

    query_centers = (
        centers.reset_index()
        .rename(
            columns={
                "item_location_id": "search_location_id",
                "latitude": "query_latitude",
                "longitude": "query_longitude",
            }
        )[["search_location_id", "query_latitude", "query_longitude"]]
    )
    fallback_centers = (
        fallback_interactions
        .dropna(subset=["search_location_id", "item_latitude", "item_longitude"])
        .groupby("search_location_id", as_index=False)
        .agg(
            fallback_latitude=("item_latitude", "median"),
            fallback_longitude=("item_longitude", "median"),
        )
    )

    queries_with_geo = (
        queries
        .merge(query_centers, on="search_location_id", how="left", validate="many_to_one")
        .merge(fallback_centers, on="search_location_id", how="left", validate="many_to_one")
    )
    queries_with_geo["query_latitude"] = queries_with_geo["query_latitude"].fillna(
        queries_with_geo["fallback_latitude"]
    )
    queries_with_geo["query_longitude"] = queries_with_geo["query_longitude"].fillna(
        queries_with_geo["fallback_longitude"]
    )
    queries_with_geo = queries_with_geo.drop(
        columns=["fallback_latitude", "fallback_longitude"]
    )

    item_centers = centers.reset_index().rename(
        columns={"latitude": "location_latitude", "longitude": "location_longitude"}
    )
    items_with_geo = items.merge(
        item_centers, on="item_location_id", how="left", validate="many_to_one"
    )
    if not np.array_equal(items_with_geo["item_id"].to_numpy(), items["item_id"].to_numpy()):
        raise ValueError("Порядок объявлений изменился при добавлении координат")

    item_latitudes = pd.to_numeric(items_with_geo["item_latitude"], errors="coerce")
    item_latitudes = item_latitudes.fillna(items_with_geo["location_latitude"])
    item_longitudes = pd.to_numeric(items_with_geo["item_longitude"], errors="coerce")
    item_longitudes = item_longitudes.fillna(items_with_geo["location_longitude"])

    query_latitudes = pd.to_numeric(queries_with_geo["query_latitude"], errors="coerce")
    query_longitudes = pd.to_numeric(queries_with_geo["query_longitude"], errors="coerce")

    return (
        queries_with_geo,
        item_latitudes.to_numpy(dtype=np.float32),
        item_longitudes.to_numpy(dtype=np.float32),
        query_latitudes.to_numpy(dtype=np.float32),
        query_longitudes.to_numpy(dtype=np.float32),
    )


def integer_ids(series):
    return series.fillna(-1).astype(np.int64).to_numpy()
