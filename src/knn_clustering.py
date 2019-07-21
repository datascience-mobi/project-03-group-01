import matplotlib.pyplot as plt
import numpy as np
import src.KNN_sklearn as KNN_sklearn
from src import pca


def get_mispredictions(train_list, test_list):
    """

    :param train_list:
    :param test_list:
    :return: None
    """
    # Runs PCA for optimal number of dimensions
    reduced_images = pca.reduce_dimensions([csv_image.image for csv_image in train_list], [csv_image.image for csv_image in test_list], 78)

    # Runs KNN for all test images and optimal k number
    predictions = KNN_sklearn.knn_sk(reduced_images[0], reduced_images[1], [csv_image.label for csv_image in train_list], 3, 0, 10000)

    # Creates table with number of misrecognitions
    table = [[0 for i in range(10)] for j in range(10)]
    for pred in predictions:
        if not pred[1] == test_list[pred[0]].label:
            table[pred[1]][test_list[pred[0]].label] += 1
    # Transforms to numpy array because it's required for plotting as heatmap
    table = np.asarray(table)

    # Plots the heatmap
    im = plt.imshow(table.reshape(10, 10), cmap='hot', interpolation='nearest')
    plt.xlabel("Correct label")
    plt.ylabel("False recognition")
    plt.title(f"Number of confusions of digit x with digit y")
    plt.colorbar(im)
    plt.show()


