import matplotlib.pyplot as plt


def display_images(images: list, figsize=(20, 20), cmap=None):
    """
    Displays n images from a list of numpy arrays representing images.

    Args:
        images (list): A list of numpy arrays representing images.
        n (int): Number of images to display.

    Returns:
        None
    """
    # Check if n is greater than the number of images
    n = len(images)
    fig, ax = plt.subplots(ncols=n)
    fig.set_size_inches(figsize)

    # Display n images using matplotlib
    for i in range(n):
        ax[i].imshow(images[i], cmap=cmap)

    plt.show()
