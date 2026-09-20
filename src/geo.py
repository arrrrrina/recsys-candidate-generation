import numpy as np

EARTH_RADIUS_KM = 6371.0


def build_location_centers(items):
    # Находит медианные координаты для каждого location_id.
    centers = items.dropna(subset=["item_location_id", "item_latitude", "item_longitude"])
    return centers.groupby("item_location_id").agg(
        latitude=("item_latitude", "median"),
        longitude=("item_longitude", "median"),
    )


def haversine_distance(latitude_1, longitude_1, latitude_2, longitude_2):
    # Вычисляет расстояние между GPS-координатами в километрах.
    latitude_1 = np.radians(latitude_1)
    longitude_1 = np.radians(longitude_1)
    latitude_2 = np.radians(latitude_2)
    longitude_2 = np.radians(longitude_2)

    latitude_difference = latitude_2 - latitude_1
    longitude_difference = longitude_2 - longitude_1

    a = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(latitude_1)
        * np.cos(latitude_2)
        * np.sin(longitude_difference / 2) ** 2
    )
    a = np.clip(a, 0, 1)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
