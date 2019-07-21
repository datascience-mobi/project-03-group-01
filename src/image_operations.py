import numpy as np
import matplotlib.pyplot as plt


def transform(data) -> np.ndarray:
    """
    Transforms list of intensity values into 28*28 numpy array
    :param data: list of intensity values for one image
    :return: 28*28 numpy array of intensity values
    """
    # Transform input list to numpy array
    data = np.asarray(data, dtype=np.uint8)

    # Transform 1D numpy array of 784 elements to 2D numpy array of 28×28 elements
    data = data.reshape(28, 28)
    return data


def draw(data) -> None:
    """
    Displays 28*28 intensity numpy array
    :param data: list of integer values from 0-255
    :return: None
    """
    # Transform to 28×28 numpy array
    data = transform(data)

    # Set as grayscale / only one pixel value per pixel
    plt.gray()
    plt.imshow(data)
    plt.show()


def save(path, data) -> None:
    """
    Saves image list to file
    :param path: path to save it to, should end on ".png"
    :param data: image list
    :return: None
    """
    # Transform to 28×28 numpy array
    data = transform(data)
    plt.imsave(path, data)
