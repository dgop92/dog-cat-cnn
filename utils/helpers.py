import pickle
from pathlib import Path


def unpickle(file_name: str, encoding="bytes"):
    with open(file_name, "rb") as fo:
        data = pickle.load(fo, encoding=encoding)
    return data


def save_to_pickle(filename: str, data: dict):
    with open(filename, "wb") as fo:
        pickle.dump(data, fo)


def get_path_from_cwd(path: str):
    return Path.cwd() / path
