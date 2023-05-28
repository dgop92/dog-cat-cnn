import logging
import os

import h5py
import numpy as np

from utils.image_algorithms import image_to_gray_scale, normalize_pixels

logger = logging.getLogger(__name__)


def create_processed_data(raw_input_path: str, processed_output_path: str):
    os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)

    with h5py.File(raw_input_path, "r") as raw_hdf5_file:
        images = raw_hdf5_file["images"][()]
        labels = raw_hdf5_file["labels"][()]
        images_to_gray_scale = map(image_to_gray_scale, images)
        normalized_images = map(normalize_pixels, images_to_gray_scale)

    logger.info("Applying transformations to images")
    new_images = np.array(list(normalized_images))

    logger.info("Saving processed images to hdf5 file")
    os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)
    with h5py.File(processed_output_path, "w") as processed_hdf5_file:
        processed_hdf5_file.create_dataset(
            "images",
            data=new_images,
            chunks=True,
            dtype="f4",
        )
        processed_hdf5_file.create_dataset(
            "labels",
            data=labels,
            chunks=True,
            dtype="i8",
        )
        processed_hdf5_file.attrs["number_of_images"] = len(new_images)


def create_processed_data_by_parts(
    raw_input_path: str, processed_output_path: str, batch_factor=0.25
):
    os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)

    with h5py.File(raw_input_path, "r") as raw_hdf5_file:
        number_of_images = int(raw_hdf5_file.attrs["number_of_images"])
        batch_size = int(number_of_images * batch_factor)
        # number_of_batches = int(number_of_images / batch_size)
        for batch_start in range(0, number_of_images, batch_size):
            logger.info(f"Filtering batch {batch_start} to {batch_start + batch_size}")
            images = raw_hdf5_file["images"][batch_start : batch_start + batch_size]
            labels = raw_hdf5_file["labels"][batch_start : batch_start + batch_size]

            images_to_gray_scale = map(image_to_gray_scale, images)
            normalized_images = map(normalize_pixels, images_to_gray_scale)
            final_images = np.array(list(normalized_images))

            logger.info("Saving processed images to hdf5 file")

            if batch_start == 0:
                logger.info("Creating hdf5 file")
                with h5py.File(processed_output_path, "w") as f:
                    f.create_dataset(
                        "images",
                        data=final_images,
                        chunks=True,
                        maxshape=(None, 32, 32, 1),
                        dtype="f4",
                    )
                    f.create_dataset(
                        "labels",
                        data=labels,
                        chunks=True,
                        maxshape=(None,),
                        dtype="i8",
                    )
            else:
                with h5py.File(processed_output_path, "a") as f:
                    f["images"].resize(
                        (f["images"].shape[0] + final_images.shape[0]), axis=0
                    )
                    f["images"][-final_images.shape[0] :] = final_images
                    f["labels"].resize((f["labels"].shape[0] + labels.shape[0]), axis=0)
                    f["labels"][-labels.shape[0] :] = labels

    logger.info("Saving number of images to hdf5 file")
    with h5py.File(processed_output_path, "a") as f:
        f.attrs["number_of_images"] = number_of_images
