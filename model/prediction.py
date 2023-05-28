import logging
from typing import List

import numpy as np
import tensorflow as tf

from utils.image_algorithms import image_to_gray_scale, normalize_pixels

logger = logging.getLogger(__name__)


def make_predicions(model_path: str, image_paths: List[str]) -> None:
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    image_matrices = []
    for image_path in image_paths:
        logger.info(f"Loading image from {image_path}")
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
        input_arr = tf.keras.preprocessing.image.img_to_array(image).astype(np.uint8)

        logger.info(f"Applying image transformations to image {image_path}")
        input_arr_gray = image_to_gray_scale(input_arr)
        input_arr_normalized = normalize_pixels(input_arr_gray)
        image_matrices.append(input_arr_normalized)

    model_input = np.array(image_matrices)
    logger.info("Making prediction")
    predictions = model.predict(model_input)

    # remember 1=dog, 0=cat

    for image_path, prediction in zip(image_paths, predictions):
        logger.info(
            f"Image {image_path} has a probability of {prediction} of being a dog"
        )
