from utils.helpers import get_path_from_cwd

RAW_IMAGE_HDF5_PATH = get_path_from_cwd("data/raw/images.hdf5")
AUGMENTED_RAW_IMAGE_HDF5_PATH = get_path_from_cwd("data/raw/augmented_images.hdf5")
PROCESSED_IMAGE_HDF5_PATH = get_path_from_cwd("data/processed/images.hdf5")

IMAGES_TRAIN_HDF5_PATH = get_path_from_cwd("data/processed/images_train.hdf5")
IMAGES_VAL_HDF5_PATH = get_path_from_cwd("data/processed/images_val.hdf5")
IMAGES_TEST_HDF5_PATH = get_path_from_cwd("data/processed/images_test.hdf5")

MODELS_PATH = get_path_from_cwd("data/models")
