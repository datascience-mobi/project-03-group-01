import matplotlib.pyplot as plt
import numpy as np


def plot_k_values(input_list):
    # plots a list in the form of [[k1, accuracy], [k2, accuracy], ...] as a diagram

    labels, ys = zip(*input_list)
    xs = np.arange(len(labels))
    width = 0.8
    plt.bar(xs, ys, width, align='center')
    plt.xticks(xs, labels)
    plt.yticks(ys)

    plt.ylabel('Accuracy')
    plt.xlabel('#k')
    plt.title('Accuracy test')
    plt.ylim(0.9, 1)  # limit y axis to see differences
    plt.show()


def plot_pca_variance(input_list):
    # plots a covariance matrix as a graph

    plt.ylabel('% Variance Explained')
    plt.xlabel('Dimensions')
    plt.title('PCA analysis')
    plt.ylim(10, 100.5)
    plt.xlim(-10, 784)
    plt.plot(input_list)
    plt.show()


if __name__ == '__main__':
    # for testing purposes
    a = [[1, 0.96], [2, 0.99], [3, 0.98], [4, 0.99], [5, 0.95]]
    plot_k_values(a)
    plot_pca_variance(a)
