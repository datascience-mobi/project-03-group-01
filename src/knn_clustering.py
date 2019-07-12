import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import src.KNN_sklearn as KNN_sklearn


def get_mispredictions(train_list, test_list):
    predictions = KNN_sklearn.knn_sk(train_list, test_list, 3, 0, 10000)
    table = [[0 for i in range(10)] for j in range(10)]
    for pred in predictions:
        if not pred[1] == test_list[pred[0]].label:
            table[pred[1]][test_list[pred[0]].label] += 1
    print(table)
    table = np.asarray(table)
    plt.imshow(table.reshape(10, 10), cmap='hot', interpolation='nearest')
    plt.xlabel("Correct label")
    plt.ylabel("False recognition")
    plt.title(f"Number of confusions of digit x with digit y")
    plt.show()


def plot_tree():
    X = np.array([[5,3],
        [10,15],
        [15,12],
        [24,10],
        [30,30],
        [85,70],
        [71,80],
        [60,78],
        [70,55],
        [80,91],])

    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(X)

    print(cluster.labels_)

    plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
    plt.show()
    from scipy.cluster.hierarchy import dendrogram, linkage

    Z = linkage(X, 'ward')

    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()


if __name__ == '__main__':
    plot_tree()