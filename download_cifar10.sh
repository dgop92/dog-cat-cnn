#!/bin/bash

# Create data folder if it doesn't exist
mkdir -p ./data

# check if data/cifar-10-batches-py does not exists
if [ ! -d "./data/cifar-10-batches-py" ]; then
    # Download cifar10 dataset
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O ./data/cifar-10-python.tar.gz

    # Extract cifar10 dataset
    tar -xvzf ./data/cifar-10-python.tar.gz -C ./data
    rm ./data/cifar-10-python.tar.gz
else
    echo "Cifar10 dataset already exists."
fi