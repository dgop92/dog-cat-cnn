from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.python.keras.models import Sequential


def create_cnn_model(input_shape: tuple) -> Sequential:
    """
    Create a convolutional neural network (CNN) model using Keras.

    Args:
        input_shape (tuple): The input shape of the images in the format (height, width, channels).

    Returns:
        Sequential: A compiled CNN model.
    """
    model = Sequential()

    # Add convolutional layers
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    # Add flatten and dense layers
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return model
