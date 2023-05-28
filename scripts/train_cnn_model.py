import argparse
import datetime

import matplotlib

from config.logging import config_logger
from config.settings import IMAGES_TRAIN_HDF5_PATH, IMAGES_VAL_HDF5_PATH
from model.train_model import train_cnn_model_in_memory, train_cnn_model_using_generator


def get_model_id_based_on_date():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


if __name__ == "__main__":
    config_logger()
    matplotlib.use("Agg")
    matplotlib.pyplot.set_loglevel("warning")
    parser = argparse.ArgumentParser(description="Train a CNN model")

    parser.add_argument(
        "--in-memory",
        type=bool,
        default=True,
        help="Train the model in memory loading all the data from the HDF5 files. if False the model will be trained using a generator to load the data from the HDF5 files. (default: True)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to train.",
        default=f"mycnn--{get_model_id_based_on_date()}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model (default: 10)",
    )
    args = parser.parse_args()

    if args.in_memory:
        train_cnn_model_in_memory(
            args.model_name, IMAGES_TRAIN_HDF5_PATH, IMAGES_VAL_HDF5_PATH, args.epochs
        )
    else:
        train_cnn_model_using_generator(
            args.model_name, IMAGES_TRAIN_HDF5_PATH, IMAGES_VAL_HDF5_PATH, args.epochs
        )
