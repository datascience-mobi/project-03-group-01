from typing import List
import numpy as np
import matplotlib.pyplot as plt


def transform(data) -> np.ndarray:
    """
    Transforms list of intensity values into 28*28 numpy array
    :param data: list of intensity values for one image
    :return: 28*28 numpy array of intensity values
    """
    data = np.asarray(data, dtype=np.uint8)
    data = data.reshape(28, 28)
    return data


def draw(data) -> None:
    """
    Displays 28*28 intensity numpy array
    :param data: intensity value list
    :return: None
    """
    data = transform(data)
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
    data = transform(data)
    plt.imsave(path, data)
