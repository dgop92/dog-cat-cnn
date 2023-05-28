import argparse

from config.logging import config_logger
from config.settings import RAW_IMAGE_HDF5_PATH
from data_pipeline.extract_dogs_and_cats import (
    create_raw_data_from_cifar_batches,
    create_raw_data_from_cifar_batches_by_parts,
)

BATCHS_PATH_FILE = "data/cifar-10-batches-py/data_batch_{batch_number}"

if __name__ == "__main__":
    config_logger()
    parser = argparse.ArgumentParser(
        description="Extract data from cifar-10 batches and save them to hdf5 file"
    )
    parser.add_argument(
        "--by-parts",
        type=bool,
        default=False,
        help="Extract the data loading just one batch into memory at a time.",
    )
    args = parser.parse_args()

    batch_path_files = [
        BATCHS_PATH_FILE.format(batch_number=batch_number)
        for batch_number in range(1, 6)
    ]

    if args.by_parts:
        create_raw_data_from_cifar_batches_by_parts(
            batch_path_files=batch_path_files, output_path=RAW_IMAGE_HDF5_PATH
        )
    else:
        create_raw_data_from_cifar_batches(
            batch_path_files=batch_path_files, output_path=RAW_IMAGE_HDF5_PATH
        )
