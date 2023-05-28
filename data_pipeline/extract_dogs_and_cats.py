import logging
import os
from typing import List, Tuple

import h5py
import numpy as np

from utils.helpers import unpickle
from utils.image_algorithms import image_row_to_matrix

logger = logging.getLogger(__name__)


def filter_images(data: np.ndarray, labels: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters the images in the given data array based on the labels.
    The label 3 is cat and 5 is dog.

    Args:
        data (np.ndarray): A Nx3072 numpy array representing images.
        labels (list): A list of N labels corresponding to the images.

    Returns:
        tuple: A tuple containing the filtered data and labels.
    """
    # Convert labels to numpy array for easier indexing
    labels = np.array(labels)

    # Find indices where label is 3 or 5
    indices = np.where((labels == 3) | (labels == 5))

    # Filter data array based on the indices
    filtered_data = data[indices]

    # Divide by 5 so that the labels are 0 and 1, now 1 is dog and 0 is cat
    new_labels = labels[indices] // 5

    return filtered_data, new_labels


def create_raw_data_from_cifar_batches(
    batch_path_files: List[str], output_path: str
) -> None:
    """
    Creates a HDF5 file containing all the images from the given batch files.
    The images are filtered so that only cats and dogs are included.

    All image are store in memory and then saved to the HDF5 file. This means
    that the batch files should not be too large.

    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    images = None
    labels = None
    number_of_images = 0

    for batch_path_file in batch_path_files:
        logger.info(f"Filtering batch {batch_path_file}")
        # Load batch
        batch_dict: dict = unpickle(batch_path_file)

        current_images = batch_dict[b"data"]
        current_labels = batch_dict[b"labels"]

        # Filter images
        filtered_data, filtered_labels = filter_images(current_images, current_labels)
        number_of_images += filtered_data.shape[0]
        images_as_matrix = np.apply_along_axis(image_row_to_matrix, 1, filtered_data)

        # extend data and labels
        if images is None:
            images = images_as_matrix
            labels = filtered_labels
        else:
            images = np.vstack((images, images_as_matrix))
            labels = np.hstack((labels, filtered_labels))

    # Save image data and labels
    logger.info(f"Saving all data with shape {images.shape}")

    with h5py.File(output_path, "w") as f:
        f.attrs["number_of_images"] = number_of_images
        f.create_dataset(
            "images",
            data=images,
            chunks=True,
            dtype="i8",
        )
        f.create_dataset(
            "labels",
            data=labels,
            chunks=True,
            dtype="i8",
        )


def create_raw_data_from_cifar_batches_by_parts(
    batch_path_files: List[str], output_path: str
):
    """
    Creates a HDF5 file containing all the images from the given batch files.
    The images are filtered so that only cats and dogs are included.

    Batches are load into memory, filtered and then directly saved to the HDF5 file.
    Memory usage is therefore limited to the size of a batch.

    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    number_of_images = 0

    for batch_number, batch_path_file in enumerate(batch_path_files, start=1):
        logger.info(f"Filtering batch {batch_path_file}")
        # Load batch
        batch_dict: dict = unpickle(batch_path_file)

        current_images = batch_dict[b"data"]
        current_labels = batch_dict[b"labels"]

        # Filter images
        filtered_data, filtered_labels = filter_images(current_images, current_labels)
        number_of_images += filtered_data.shape[0]
        images_as_matrix = np.apply_along_axis(image_row_to_matrix, 1, filtered_data)

        if batch_number == 1:
            logger.info(f"Saving first batch with {images_as_matrix.shape} shape")
            with h5py.File(output_path, "w") as f:
                f.create_dataset(
                    "images",
                    data=images_as_matrix,
                    chunks=True,
                    maxshape=(None, 32, 32, 3),
                    dtype="i8",
                )
                f.create_dataset(
                    "labels",
                    data=filtered_labels,
                    chunks=True,
                    maxshape=(None,),
                    dtype="i8",
                )
        else:
            logger.info(
                f"Saving batch {batch_number} with {images_as_matrix.shape} shape"
            )
            with h5py.File(output_path, "a") as f:
                f["images"].resize(
                    (f["images"].shape[0] + images_as_matrix.shape[0]), axis=0
                )
                f["images"][-images_as_matrix.shape[0] :] = images_as_matrix

                f["labels"].resize(
                    (f["labels"].shape[0] + filtered_labels.shape[0]), axis=0
                )
                f["labels"][-filtered_labels.shape[0] :] = filtered_labels

    with h5py.File(output_path, "a") as f:
        f.attrs["number_of_images"] = number_of_images
