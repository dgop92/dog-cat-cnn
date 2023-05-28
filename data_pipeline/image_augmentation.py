import logging
import os

import albumentations as A
import h5py
import numpy as np

logger = logging.getLogger(__name__)


def get_new_images_through_augmentation(images: np.ndarray):
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.8),
            A.Rotate(limit=10, p=0.8),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )

    logger.info("Applying transformations to images")

    # cv2 images requires uint8 dtype
    images = images.astype(np.uint8)
    # we are duplicating the images, each image will have a new version
    new_images = np.array(
        list(map(lambda image: transform(image=image)["image"], images))
    )

    return new_images


def create_agumented_raw_data(raw_input_path: str, augmented_raw_output_path: str):
    os.makedirs(os.path.dirname(augmented_raw_output_path), exist_ok=True)

    with h5py.File(raw_input_path, "r") as raw_hdf5_file:
        images = raw_hdf5_file["images"][()]
        labels = raw_hdf5_file["labels"][()]

    augmented_images = get_new_images_through_augmentation(images)
    new_images = np.concatenate((images, augmented_images))
    new_labels = np.concatenate((labels, labels))

    with h5py.File(augmented_raw_output_path, "w") as augmented_hdf5_file:
        augmented_hdf5_file.create_dataset(
            "images",
            data=new_images,
            chunks=True,
            dtype="i8",
        )
        augmented_hdf5_file.create_dataset(
            "labels",
            data=new_labels,
            chunks=True,
            dtype="i8",
        )
        augmented_hdf5_file.attrs["number_of_images"] = len(new_images)
