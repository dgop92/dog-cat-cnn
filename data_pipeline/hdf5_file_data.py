from typing import Generator, Tuple

import h5py
import numpy as np


class ImageGenerator:
    def __init__(self, path: str, batch_size: int = 32):
        self.path = path
        self.batch_size = batch_size

    def __call__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        with h5py.File(self.path, "r") as f:
            number_of_images = f.attrs["number_of_images"]
            images = f["images"]
            labels = f["labels"]

            for i in range(0, number_of_images, self.batch_size):
                batch_images = images[i : i + self.batch_size]
                batch_labels = labels[i : i + self.batch_size]
                yield batch_images, batch_labels


def get_image_data_from_hd5f(path: str) -> dict:
    with h5py.File(path, "r") as f:
        number_of_images = f.attrs["number_of_images"]
        images = f["images"][()]
        labels = f["labels"][()]
        return {
            "number_of_images": number_of_images,
            "images": images,
            "labels": labels,
        }
