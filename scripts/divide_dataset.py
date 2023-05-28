import argparse

from config.logging import config_logger
from config.settings import (
    IMAGES_TEST_HDF5_PATH,
    IMAGES_TRAIN_HDF5_PATH,
    IMAGES_VAL_HDF5_PATH,
    PROCESSED_IMAGE_HDF5_PATH,
)
from data_pipeline.data_split import split_dataset, split_dataset_by_parts

if __name__ == "__main__":
    config_logger()
    parser = argparse.ArgumentParser(description="Create processed data from raw data.")
    parser.add_argument(
        "--by-parts",
        type=bool,
        default=False,
        help="Extract the data loading just one batch into memory at a time.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="The batch size to use when reading the data. If by-parts is True, this is the batch size to use when writing the data.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Size of the test set. Must be between 0 and 1. Default: 0.2",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Size of the validation set. Must be between 0 and 1. Default: 0.2",
    )

    args = parser.parse_args()

    if args.by_parts:
        split_dataset_by_parts(
            PROCESSED_IMAGE_HDF5_PATH,
            IMAGES_TRAIN_HDF5_PATH,
            IMAGES_TEST_HDF5_PATH,
            IMAGES_VAL_HDF5_PATH,
            args.batch_size,
            args.test_size,
            args.val_size,
        )
    else:
        split_dataset(
            PROCESSED_IMAGE_HDF5_PATH,
            IMAGES_TRAIN_HDF5_PATH,
            IMAGES_TEST_HDF5_PATH,
            IMAGES_VAL_HDF5_PATH,
            args.test_size,
            args.val_size,
        )
