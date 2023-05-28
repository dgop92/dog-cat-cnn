import logging

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_dataset(
    processed_data_path: str,
    train_output_path: str,
    test_output_path: str,
    validation_output_path: str,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split the data into training, validation, and testing sets and
    save them to hdf5 files
    """
    logger.info("Loading processed data from hdf5 file")
    with h5py.File(processed_data_path, "r") as processed_hdf5_file:
        images = processed_hdf5_file["images"][:]
        labels = processed_hdf5_file["labels"][:]

    logger.info("Splitting images into training and testing sets")
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state
    )

    logger.info("Splitting training images into training and validation sets")
    images_train, images_val, labels_train, labels_val = train_test_split(
        images_train, labels_train, test_size=validation_size, random_state=random_state
    )

    logger.info("Creating hdf5 files for training, testing, and validation sets")
    with h5py.File(train_output_path, "w") as f_train, h5py.File(
        test_output_path, "w"
    ) as f_test, h5py.File(validation_output_path, "w") as f_val:
        f_train.create_dataset(
            "images",
            data=images_train,
            chunks=True,
            dtype="f4",
        )
        f_train.create_dataset(
            "labels",
            data=labels_train,
            chunks=True,
            dtype="i8",
        )
        f_train.attrs["number_of_images"] = images_train.shape[0]

        f_test.create_dataset(
            "images",
            data=images_test,
            chunks=True,
            dtype="f4",
        )
        f_test.create_dataset(
            "labels",
            data=labels_test,
            chunks=True,
            dtype="i8",
        )
        f_test.attrs["number_of_images"] = images_test.shape[0]

        f_val.create_dataset(
            "images",
            data=images_val,
            chunks=True,
            dtype="f4",
        )
        f_val.create_dataset(
            "labels",
            data=labels_val,
            chunks=True,
            dtype="i8",
        )
        f_val.attrs["number_of_images"] = images_val.shape[0]


def split_dataset_by_parts(
    processed_data_path: str,
    train_output_path: str,
    test_output_path: str,
    validation_output_path: str,
    batch_size: int = 1000,
    test_size: float = 0.2,
    validation_size: float = 0.2,
) -> tuple:
    """
    Split the data into training, validation, and testing sets and
    save them to hdf5 files

    This function is used when the data is too large to fit into memory
    """
    logger.info("Loading processed images from hdf5 file")
    with h5py.File(processed_data_path, "r") as hdf5_file:
        number_of_images = hdf5_file.attrs["number_of_images"]

        train_size = 1 - test_size - validation_size

        # Compute the number of examples in each set
        train_size = int(number_of_images * train_size)
        test_size = int(number_of_images * test_size)
        val_size = int(number_of_images * validation_size)

        logger.info("Shuffling indices")
        indices = np.arange(number_of_images)
        np.random.shuffle(indices)

        # Create the output HDF5 files
        with h5py.File(train_output_path, "w") as f_train, h5py.File(
            test_output_path, "w"
        ) as f_test, h5py.File(validation_output_path, "w") as f_val:
            logger.info(
                "Creating hdf5 files for training, testing, and validation sets"
            )
            f_train.create_dataset(
                "images",
                shape=(train_size, 32, 32, 1),
                chunks=True,
                maxshape=(None, 32, 32, 1),
                dtype="f4",
            )
            f_train.create_dataset(
                "labels", shape=(train_size,), chunks=True, maxshape=(None,), dtype="i8"
            )
            f_train.attrs["number_of_images"] = train_size

            f_test.create_dataset(
                "images",
                shape=(test_size, 32, 32, 1),
                chunks=True,
                maxshape=(None, 32, 32, 1),
                dtype="f4",
            )
            f_test.create_dataset(
                "labels", shape=(test_size,), chunks=True, maxshape=(None,), dtype="i8"
            )
            f_test.attrs["number_of_images"] = test_size

            f_val.create_dataset(
                "images",
                shape=(val_size, 32, 32, 1),
                chunks=True,
                maxshape=(None, 32, 32, 1),
                dtype="f4",
            )
            f_val.create_dataset(
                "labels", shape=(val_size,), chunks=True, maxshape=(None,), dtype="i8"
            )
            f_val.attrs["number_of_images"] = val_size

            # Copy the data from the input file to the output files
            for i in range(0, number_of_images, batch_size):
                batch_indices = np.sort(indices[i : i + batch_size])
                batch_images = hdf5_file["images"][batch_indices]
                batch_labels = hdf5_file["labels"][batch_indices]

                logger.info(f"Filtering batch {i} to {i + batch_size}")

                if i < train_size:
                    logger.info(f"Saving batch {i} to {i + batch_size} to training set")
                    f_train["images"][i : i + batch_size] = batch_images
                    f_train["labels"][i : i + batch_size] = batch_labels
                elif i < train_size + test_size:
                    logger.info(f"Saving batch {i} to {i + batch_size} to testing set")
                    j = i - train_size
                    f_test["images"][j : j + batch_size] = batch_images
                    f_test["labels"][j : j + batch_size] = batch_labels
                else:
                    logger.info(
                        f"Saving batch {i} to {i + batch_size} to validation set"
                    )
                    j = i - train_size - test_size
                    f_val["images"][j : j + batch_size] = batch_images
                    f_val["labels"][j : j + batch_size] = batch_labels
