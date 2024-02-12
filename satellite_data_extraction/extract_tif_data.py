import os
import os.path as osp
import tqdm
from typing import Any
from joblib import Parallel, delayed
from collections import ChainMap
import threading
import time
from tqdm.auto import tqdm

import multiprocessing
import numpy as np
import pandas as pd
from PIL import Image
from datacube_extractor import DataCubeExtractor, ImageDataCubeExtractor


def get_quarter_dates(
        year: int,
        quarter: int
):
    if quarter == 1:
        return "{}.12.02..{}.03.20".format(year - 1, year)
    if quarter == 2:
        return "{}.03.21..{}.06.24".format(year, year)
    if quarter == 3:
        return "{}.06.25..{}.09.12".format(year, year)
    if quarter == 4:
        return "{}.09.13..{}.12.01".format(year, year)


def create_tile_bboxes_from_df(
        df: pd.DataFrame,
        latitude_col_name: str,
        longitude_col_name: str,
        tile_size_deg: tuple[float, float]
) -> list[float, float, float, float]:
    min_latitude, max_latitude = (
        df[latitude_col_name].min(),
        df[latitude_col_name].max(),
    )
    min_longitude, max_longitude = (
        df[longitude_col_name].min(),
        df[longitude_col_name].max(),
    )
    latitude_step, longitude_step = tile_size_deg

    latitude_points = np.arange(min_latitude, max_latitude, latitude_step)
    longitude_points = np.arange(min_longitude, max_longitude, longitude_step)

    if len(latitude_points) < 1 or len(longitude_points) < 1:
        return []

    tile_bboxes = []
    for latitude in latitude_points:
        for longitude in longitude_points:
            tile_bboxes.append((latitude, longitude, latitude_step, longitude_step))

    return tile_bboxes


def get_metadata_in_tile(
        df: pd.DataFrame,
        tile_bbox: tuple[float, float, float, float],
        latitude_col: str,
        longitude_col: str
) -> pd.DataFrame:
    left, bottom, width, height = tile_bbox

    df_in_tile = df[
        ((left <= df[latitude_col]) & (df[latitude_col] < left + width))
        & ((bottom <= df[longitude_col]) & (df[longitude_col] < bottom + height))
    ]

    return df_in_tile


def search_tile(
        position_df: pd.DataFrame,
        tile_bbox: tuple,
        latitude_col_name: str,
        longitude_col_name: str,
        tile_image_output_dir: str = None
) -> dict:
    try:
        df_in_tile = get_metadata_in_tile(position_df, tile_bbox, latitude_col_name, longitude_col_name)
        if len(df_in_tile) == 0:
            return {}

        extractor = DataCubeExtractor(tile_bbox, band_index=1)
        # extractor = ImageDataCubeExtractor(tile_bbox, band_index=1, image_patch_size=64)

        has_data = extractor.load_raster(raster_path)

        if not has_data:
            print(f"No data in tile {tile_bbox}")
            return {}

        extracted_values_indexed = {}
        for index, row in df_in_tile.iterrows():
            extracted_value = extractor[row.tolist()]
            if extracted_value is None:
                continue
            extracted_values_indexed[index] = int(extracted_value)

            if tile_image_output_dir:
                extractor.save_patch_image(
                    item=row.tolist(), tile_image_output_dir=tile_image_output_dir
                )

        return extracted_values_indexed

    except Exception as e:
        print(f"Error in tile {tile_bbox}: {e}")
        return {}


if __name__ == "__main__":
    n_jobs = multiprocessing.cpu_count()

    metadata_path = "/data/zenith/share/GeoLifeCLEF/2024/PresenceAbsenceSurveys/GLC24_PA_metadata_test.csv"
    raster_data_path = (
        "/data/zenith/share/GeoLifeCLEF/2023/data/landsat/"
    )
    output_dir = "/data/zenith/share/GeoLifeCLEF/2024/SatelliteTimeSeries/"

    create_images = False
    tile_size_deg = (2, 2)
    latitude_col_name = "lat"
    longitude_col_name = "lon"
    unique_id = "surveyId"

    bands = ["red", "green", "blue", "nir", "swir1", "swir2"]

    for band in bands:
        print(f"Starting processing band: {band}")
        output_file_name = f"GLC24-PA-test-landsat_time_series-{band}"

        years = np.arange(2000, 2022)

        metadata = pd.read_csv(
            metadata_path,
            delimiter=",",
            low_memory=False,
        )

        header = []
        metadata = metadata.drop_duplicates(unique_id).reset_index(drop=True)
        out_metadata = metadata[[unique_id]].copy(deep=True)

        time_stamp = 0
        for year in years:
            preceding_yearly_metadata = metadata[year <= metadata.year]
            if len(preceding_yearly_metadata) == 0:
                print(f"No data found in {year}. Skipping!")
                continue
            else:
                print(f"Processing {len(preceding_yearly_metadata)} rows.")
            position_df = preceding_yearly_metadata[[latitude_col_name, longitude_col_name]]
            position_df = position_df.sort_values(
                by=[latitude_col_name, longitude_col_name]
            )

            tile_bboxes = create_tile_bboxes_from_df(
                position_df, latitude_col_name, longitude_col_name, tile_size_deg
            )

            for quarter in range(1, 5):
                raster_path = f"{raster_data_path}/lcv_{band}_landsat.glad.ard_p50_30m_0..0cm_{get_quarter_dates(year=year, quarter=quarter)}_eumap_epsg3035_v1.1.tif"

                if not os.path.isfile(raster_path):
                    print(f"Raster for q {quarter} of {year} not found. Skipping!")
                    continue
                print(f"Extraction for q {quarter} of {year} started!")

                tile_image_output_dir = None
                if create_images:
                    tile_image_output_dir = f"{output_dir}/images/{year}_{quarter}"
                    os.makedirs(tile_image_output_dir, exist_ok=True)

                start_time = time.time()
                extracted_tiles = Parallel(n_jobs=n_jobs)(
                    delayed(search_tile)(position_df, tile_bbox, latitude_col_name, longitude_col_name, tile_image_output_dir)
                    for tile_bbox in tqdm(tile_bboxes, total=len(tile_bboxes))
                )
                extracted_values_indexed = dict(ChainMap(*extracted_tiles))

                out_metadata[f"{year}_{quarter}"] = out_metadata.index.map(
                    extracted_values_indexed
                )
                print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
                found_values_total = sum(~out_metadata[f"{year}_{quarter}"].isna())
                print(f"Found: {found_values_total} values from initial: {len(preceding_yearly_metadata)} entries "
                      f"-> {found_values_total/len(preceding_yearly_metadata):.2%}")

        out_metadata.to_csv(f"{output_dir}/{output_file_name}.csv", index=False)
        print(f"Band: {band} done!")
