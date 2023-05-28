import argparse

from config.logging import config_logger
from config.settings import (
    AUGMENTED_RAW_IMAGE_HDF5_PATH,
    PROCESSED_IMAGE_HDF5_PATH,
    RAW_IMAGE_HDF5_PATH,
)
from data_pipeline.transformations import (
    create_processed_data,
    create_processed_data_by_parts,
)

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
        "--batch-factor",
        type=float,
        default=0.25,
        help="factor to compute the batch size. The batch size will be the factor times the number of images.",
    )
    parser.add_argument(
        "--augment",
        type=bool,
        default=False,
        help="if True, data will be read from AUGMENTED_RAW_IMAGE_HDF5_PATH, otherwise from RAW_IMAGE_HDF5_PATH",
    )
    args = parser.parse_args()

    if args.by_parts:
        create_processed_data_by_parts(
            AUGMENTED_RAW_IMAGE_HDF5_PATH if args.augment else RAW_IMAGE_HDF5_PATH,
            PROCESSED_IMAGE_HDF5_PATH,
            args.batch_factor,
        )
    else:
        create_processed_data(
            AUGMENTED_RAW_IMAGE_HDF5_PATH if args.augment else RAW_IMAGE_HDF5_PATH,
            PROCESSED_IMAGE_HDF5_PATH,
        )
