import logging

import tensorflow as tf

from data_pipeline.hdf5_file_data import ImageGenerator, get_image_data_from_hd5f
from model.cnn_model import create_cnn_model
from model.model_persistence import save_model, save_model_metrics

logger = logging.getLogger(__name__)


def train_cnn_model_in_memory(
    model_name: str,
    train_output_path: str,
    validation_output_path: str,
    epochs: int = 10,
) -> None:
    train_images_data = get_image_data_from_hd5f(train_output_path)
    val_images_data = get_image_data_from_hd5f(validation_output_path)

    train_images = train_images_data["images"]
    train_labels = train_images_data["labels"]

    val_images = val_images_data["images"]
    val_labels = val_images_data["labels"]

    logger.info(f"{model_name} - Shape of train images: {train_images.shape}")
    logger.info(f"{model_name} - Shape of train labels: {train_labels.shape}")
    logger.info(f"{model_name} - Shape of val images: {val_images.shape}")
    logger.info(f"{model_name} - Shape of val labels: {val_labels.shape}")

    mycnn = create_cnn_model(input_shape=(32, 32, 1))
    mycnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    tensor_board_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"logs/{model_name}", histogram_freq=1
    )

    logger.info(f"{model_name} - Training model with {epochs} epochs...")
    history = mycnn.fit(
        train_images,
        train_labels,
        epochs=epochs,
        validation_data=(val_images, val_labels),
        verbose=0,
        callbacks=[tensor_board_callback],
        batch_size=32,
    )

    save_model_metrics(history, model_name)
    save_model(mycnn, model_name)


def train_cnn_model_using_generator(
    model_name: str,
    train_output_path: str,
    validation_output_path: str,
    epochs: int = 10,
) -> None:
    train_img_generator = ImageGenerator(train_output_path, 32)
    val_img_generator = ImageGenerator(validation_output_path, 32)

    train_dataset = tf.data.Dataset.from_generator(
        train_img_generator,
        output_signature=(
            # bath_size, height, width, channels
            tf.TensorSpec(shape=(None, 32, 32, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
        ),
    )
    val_dataset = tf.data.Dataset.from_generator(
        val_img_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 32, 32, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
        ),
    )

    mycnn = create_cnn_model(input_shape=(32, 32, 1))
    mycnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    tensor_board_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"logs/{model_name}", histogram_freq=1
    )

    logger.info(f"{model_name} - Training model with {epochs} epochs...")
    history = mycnn.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        verbose=0,
        callbacks=[tensor_board_callback],
        batch_size=32,
    )
    save_model_metrics(history, model_name)
    save_model(mycnn, model_name)
