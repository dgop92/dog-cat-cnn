import argparse

from config.logging import config_logger
from config.settings import AUGMENTED_RAW_IMAGE_HDF5_PATH, RAW_IMAGE_HDF5_PATH
from data_pipeline.image_augmentation import create_agumented_raw_data

if __name__ == "__main__":
    config_logger()
    parser = argparse.ArgumentParser(
        description="Create augmented raw data from raw data."
    )
    parser.add_argument(
        "--by-parts",
        type=bool,
        default=False,
        help="Extract the data loading just one batch into memory at a time.",
    )
    parser.add_argument(
        "--batch-factor",
        type=float,
        default=0.25,
        help="factor to compute the batch size. The batch size will be the factor times the number of images.",
    )
    args = parser.parse_args()

    if args.by_parts:
        raise NotImplementedError("By parts not implemented for augmented data")
    else:
        create_agumented_raw_data(
            RAW_IMAGE_HDF5_PATH,
            AUGMENTED_RAW_IMAGE_HDF5_PATH,
        )
