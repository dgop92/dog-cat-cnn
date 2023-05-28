import logging
import os

import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from config.settings import MODELS_PATH

logger = logging.getLogger(__name__)


def save_epochs_vs_accuracy(history: tf.keras.callbacks.History, path: str) -> None:
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    # set size of the figure
    plt.rcParams["figure.figsize"] = [10, 5]
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig(path)
    plt.clf()


def save_epochs_vs_loss(history: tf.keras.callbacks.History, path: str) -> None:
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    # set size of the figure
    plt.rcParams["figure.figsize"] = [10, 5]
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig(path)
    plt.clf()


def save_history_as_csv(history: tf.keras.callbacks.History, model_name: str) -> None:
    current_model_path = os.path.join(MODELS_PATH, model_name)
    os.makedirs(current_model_path, exist_ok=True)

    logger.info(f"Saving history as csv for {model_name}...")
    pd.DataFrame(history.history).to_csv(
        os.path.join(current_model_path, "history.csv"), index=False
    )


def save_model_metrics(history: tf.keras.callbacks.History, model_name: str) -> None:
    current_model_path = os.path.join(MODELS_PATH, model_name)
    os.makedirs(current_model_path, exist_ok=True)

    logger.info(f"Saving accuracy vs epochs picture for {model_name}...")
    save_epochs_vs_accuracy(history, os.path.join(current_model_path, "accuracy.png"))
    logger.info(f"Saving loss vs epochs picture for {model_name}...")
    save_epochs_vs_loss(history, os.path.join(current_model_path, "loss.png"))
    logger.info(f"Saving history as csv for {model_name}...")
    save_history_as_csv(history, model_name)


def save_model(model: tf.keras.Model, model_name: str) -> None:
    current_model_path = os.path.join(MODELS_PATH, model_name)
    os.makedirs(current_model_path, exist_ok=True)

    logger.info(f"Saving model {model_name}...")
    model.save(os.path.join(current_model_path, "model.h5"))
