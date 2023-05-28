import numpy as np


def image_row_to_matrix(image: np.ndarray) -> np.ndarray:
    """Converts a row of 3072 pixels to a 32x32x3 matrix."""

    r_channel = image[:1024].reshape((32, 32))
    g_channel = image[1024:2048].reshape((32, 32))
    b_channel = image[2048:].reshape((32, 32))

    return np.stack((r_channel, g_channel, b_channel), axis=2)


def image_to_gray_scale(image: np.ndarray) -> np.ndarray:
    """Converts a 32x32x3 RGB image to a 32x32x1 gray scale image"""

    return np.dot(image[..., :3], [0.299, 0.587, 0.114]).reshape((32, 32, 1))


def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """Normalizes the pixels of an image to be between 0 and 1"""

    return image / 255.0


def de_normalize_pixels(image: np.ndarray) -> np.ndarray:
    """De-normalizes the pixels of an image to be between 0 and 255"""

    return image * 255.0
