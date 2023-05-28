import argparse

from config.logging import config_logger
from model.prediction import make_predicions

if __name__ == "__main__":
    config_logger()
    parser = argparse.ArgumentParser(description="Make predictions using a CNN model")

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model to use to make the prediction.",
    )

    parser.add_argument(
        "image_paths",
        type=str,
        help="List of paths to the images to use to make the prediction. The paths must be separated by a comma.",
    )

    args = parser.parse_args()

    image_paths = args.image_paths.split(",")
    make_predicions(args.model_path, image_paths)
