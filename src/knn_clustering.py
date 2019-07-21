import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import src.KNN_sklearn as KNN_sklearn
from src import pca


def get_mispredictions(train_list, test_list):
    reds = pca.reduce_dimensions([csv_image.image for csv_image in train_list], [csv_image.image for csv_image in test_list], 78)

    predictions = KNN_sklearn.knn_sk(reds[0], reds[1], [csv_image.label for csv_image in train_list], 3, 0, 10000)
    table = [[0 for i in range(10)] for j in range(10)]
    for pred in predictions:
        if not pred[1] == test_list[pred[0]].label:
            table[pred[1]][test_list[pred[0]].label] += 1
    table = np.asarray(table)
    im = plt.imshow(table.reshape(10, 10), cmap='hot', interpolation='nearest')
    plt.xlabel("Correct label")
    plt.ylabel("False recognition")
    plt.title(f"Number of confusions of digit x with digit y")
    plt.colorbar(im)
    plt.show()


