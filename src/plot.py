import src.pickle_operations as pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

import src.KNN_sklearn as KNN_sklearn
import src.load_image_vectors as load_image_vectors

total_number = 0
success_number = 0
k_accuracy = list()


def reset_rates():
    global total_number, success_number
    success_number = 0
    total_number = 0


def get_success_rate():
    global total_number, success_number
    success = float(success_number) / float(total_number)
    return round(success, 4)


def create_sklearn_k_accuracy_list(k_min, k_max):
    # creates a list in the form of [[k1, accuracy], [k2, accuracy], ...]
    global total_number, success_number
    global k_accuracy
    print("Started sklearn_k_value_test")
    for k in range(k_min, k_max):
        prediction_list = KNN_sklearn.knn_sk(test_lists, training_lists, k, 10)
        for idx, prediction in enumerate(prediction_list):
            total_number += 1
            if prediction == test_lists[0][idx].label:
                success_number += 1
        k_accuracy.append([k, get_success_rate()])
        reset_rates()
        print("Finished accuracy calculation " + str(k))


def pca_variance_analysis(input_list):
    # calculates the retained variance for different dimensions and plots the resulting variance-matrix
    print("Started pca_variance_analysis")
    scale = preprocessing.StandardScaler()
    scale.fit(input_list)
    input_list = scale.transform(input_list)
    covar_matrix = PCA(n_components=784)  # we have 784 features
    covar_matrix.fit(input_list)
    variance = covar_matrix.explained_variance_ratio_  # calculate variance ratios
    var = np.cumsum(np.round(variance, decimals=5) * 100)
    print(var)  # cumulative sum of variance explained with [n] features
    plot_pca_variance(var)


# def k_value_test(k_min, k_max):

# creates a list in the form of [[k1, accuracy], [k2, accuracy], ...]
# then plots this created list as a diagram

# k_accuracy = list()
# for k in range(k_min, k_max):
# for i in range(10, 100):
# sorted_distances = knn.get_sorted_distances(test_list[i], training_list)
# predicted_digit = knn.knn_distance_prediction(sorted_distances, k)
# knn.set_success_rate(predicted_digit, test_list[i])
# k_accuracy.append([k, knn.get_success_rate()])
# knn.reset_rates()
# plot.plot_k_values(k_accuracy)


def save_results():
    plt.savefig("sklearn_k_vale_test.png")  # save the figure to file as png
    plt.savefig("sklearn_k_vale_test.pdf")  # save the figure to file as pdf


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
    plt.ylim(0.95, 1)  # limit y axis to see differences
    save_results()
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
    # a = [[1, 0.96], [2, 0.99], [3, 0.98], [4, 0.99], [5, 0.95]]
    # plot_k_values(a)

    # load training and test images
    training_lists = load_image_vectors.load_gz('../data/mnist_train.csv.gz')
    print("Successfully loaded training list")
    test_lists = load_image_vectors.load_gz('../data/mnist_test.csv.gz')
    print("Successfully loaded test list")

    create_sklearn_k_accuracy_list(1, 4)
    pickle.save_pickles(k_accuracy, "k_accuracy.dat")
    plot_k_values(pickle.load_pickles("k_accuracy.dat"))
    # pca_variance_analysis(test_lists[1])
