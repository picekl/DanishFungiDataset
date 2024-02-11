import os.path as osp
import numpy as np
import pyproj
import rasterio
from PIL import Image


class DataCubeExtractor:
    def __init__(self, tile_bbox: tuple, band_index: int = 1, padding: float = 0.0):
        self.tile_bbox = tile_bbox
        self.band_index = band_index
        self.padding = padding

        # From loaded raster
        self.transformer = None
        self.tile_data = None

        self.x_resolution, self.y_resolution = None, None
        self.left, self.bottom = None, None
        self.right, self.top = None, None

    def load_raster(self, raster_path: str) -> bool:
        """Returns 'tile has data'"""

        with rasterio.open(raster_path, "r") as tif_file:
            self.transformer = pyproj.Transformer.from_crs(
                "epsg:4326", tif_file.crs, always_xy=True
            )  # xx = lon, yy = lat

            self.x_resolution, self.y_resolution = tif_file.res[:2]

            self.left, self.bottom = self.transformer.transform(
                yy=self.tile_bbox[0] - self.padding, xx=self.tile_bbox[1] - self.padding
            )
            self.right, self.top = self.transformer.transform(
                yy=self.tile_bbox[0] + self.tile_bbox[2] + self.padding,
                xx=self.tile_bbox[1] + self.tile_bbox[3] + self.padding,
            )

            tile_window = rasterio.windows.from_bounds(
                self.left,
                self.bottom,
                self.right,
                self.top,
                transform=tif_file.transform,
            )
            self.tile_data = tif_file.read(self.band_index, window=tile_window)

            return self.tile_data.shape[0] > 0 and self.tile_data.shape[1] > 0

    def __getitem__(self, item: tuple[float, float]) -> int:
        """
        :param item: the GPS location (latitude, longitude)
        :return: value
        """

        # convert the lat, lon coordinates to raster EPSG.
        x_index, y_index = self._item_to_tile_index(
            item
        )  # x: left->right, y: top->bottom

        if x_index and y_index:
            if self.tile_data[y_index, x_index] == 0:
                return self._search_radius_mean_indexed(
                    x_index, y_index, radius_index=1
                )

            return int(self.tile_data[y_index, x_index])

        else:
            # print(f"Item {item} not in tile {self.tile_bbox}. Increase padding?")
            return None

    def _item_to_tile_index(self, item: tuple[float, float]):
        item_x, item_y = self.transformer.transform(yy=item[0], xx=item[1])

        if self.left <= item_x <= self.right and self.bottom <= item_y <= self.top:
            x_index = int((item_x - self.left) / self.x_resolution)
            y_index = int(
                self.tile_data.shape[0] - (item_y - self.bottom) / self.y_resolution
            )

            # x_index = int((item_x - self.left) / (self.right - self.left) * self.tile_data.shape[1])
            # y_index = int((item_y - self.bottom) / (self.top - self.bottom) * self.tile_data.shape[0])

            return x_index, y_index
        return None, None

    def _search_radius_mean_indexed(
        self, x_index: int, y_index: int, radius_index: int = 1
    ) -> float:
        """Search around the given index for the max value. Returns 0 if no value is found."""
        try:
            return self.tile_data[
                y_index - radius_index : y_index + radius_index,
                x_index - radius_index : x_index + radius_index,
            ].mean()
        except Exception as e:
            print(e)
            return 0


class ImageDataCubeExtractor(DataCubeExtractor):
    def __init__(
        self,
        tile_bbox: tuple,
        band_index: int = 1,
        padding: float = 0.5,
        convert_to_uint8: bool = False,
        gamma_for_conversion: float = 2.5,
        image_patch_size: int = 128,
    ):
        super().__init__(tile_bbox, band_index=band_index, padding=padding)

        self.convert_to_uint8 = convert_to_uint8
        self.gamma_for_conversion = gamma_for_conversion
        self.image_patch_size = image_patch_size

    def convert_tile_data_to_uint8(self, tile_data: np.ndarray) -> np.ndarray:
        """??? Imported from previous project -> Likely used for image creation"""
        tile_data = np.clip(tile_data / 10000.0, a_min=0, a_max=1.0)
        tile_data = (tile_data ** (1 / self.gamma_for_conversion)) * 256
        tile_data = tile_data.astype(np.uint8)
        # (tile_data - tile_data.min()) / (tile_data.max() - tile_data.min()) * 255
        return tile_data

    def save_tile_image(self, tile_image_output_dir: str) -> None:
        """Saves whole tile"""
        assert osp.isdir(
            tile_image_output_dir
        ), f"Photo output directory '{tile_image_output_dir}' does not exist!"
        assert self.tile_data is not None, "Tile data not loaded!"

        start_latitude = self.tile_bbox[0]
        start_longitude = self.tile_bbox[1]

        tile_data = self.tile_data.astype(np.uint8)

        image = Image.fromarray(
            tile_data,
            "L",
        )
        image_path = osp.join(
            tile_image_output_dir,
            f"lat-({start_latitude:+03.1f})_lon-({start_longitude:+03.1f}).jpeg",
        )
        image.save(image_path)

    def save_patch_image(self, item, tile_image_output_dir: str) -> str:
        """Saves patch surrounding the item"""
        assert osp.isdir(
            tile_image_output_dir
        ), f"Photo output directory '{tile_image_output_dir}' does not exist!"
        assert self.tile_data is not None, "Tile data not loaded!"

        center_latitude, center_longitude = item
        index_x, index_y = self._item_to_tile_index(item)
        patch_tile = self._select_image_patch(index_x, index_y, self.image_patch_size)

        if self.convert_to_uint8:
            patch_tile = self.convert_tile_data_to_uint8(patch_tile)

        image = Image.fromarray(
            patch_tile,
            "L",
        )
        image_path = osp.join(
            tile_image_output_dir,
            f"lat-({center_latitude:0=+011.6f})_lon-({center_longitude:0=+011.6f}).jpeg",
        )
        image.save(image_path)
        return image_path

    def _select_image_patch(
        self, index_x: int, index_y: int, patch_size: int
    ) -> np.ndarray:
        left_x = max(index_x - patch_size // 2, 0)
        right_x = min(index_x + patch_size // 2, self.tile_data.shape[1])
        top_y = max(index_y - patch_size // 2, 0)
        bottom_y = min(index_y + patch_size // 2, self.tile_data.shape[0])

        patch_tile = self.tile_data[top_y:bottom_y, left_x:right_x]

        if (
            self.image_patch_size != patch_tile.shape[0]
            or self.image_patch_size != patch_tile.shape[1]
        ):
            print(
                f"Cropped image {patch_tile.shape} does not match patch size {self.image_patch_size}! Increase padding?"
            )

        return patch_tile
