import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

import src.KNN_sklearn as KNN_sklearn
import src.load_image_vectors as load_image_vectors
import src.plot as plot

total_number = 0
success_number = 0


def get_success_rate():
    global total_number, success_number
    success = float(success_number) / float(total_number)
    return round(success, 4)


def sklearn_k_value_test(k_min, k_max):
    # creates a list in the form of [[k1, accuracy], [k2, accuracy], ...]
    # then plots this created list as a diagram
    global total_number, success_number
    k_accuracy = list()
    for k in range(k_min, k_max):
        prediction_list = KNN_sklearn.knn_sk(test_lists, training_lists, k, 10)
        for idx, prediction in enumerate(prediction_list):
            total_number += 1
            if prediction == test_lists[0][idx].label:
                success_number += 1
        print("Success rate = " + str(get_success_rate))
        k_accuracy.append([k, get_success_rate()])
        reset_rates()
    plot.plot_k_values(k_accuracy)


def reset_rates():
    global total_number, success_number
    success_number = 0
    total_number = 0


def pca_variance_analysis(input_list):
    # calculates the retained variance for different dimensions and plots the resulting variance-matrix

    scale = preprocessing.StandardScaler()
    scale.fit(input_list)
    input_list = scale.transform(input_list)
    covar_matrix = PCA(n_components=784)  # we have 784 features
    covar_matrix.fit(input_list)
    variance = covar_matrix.explained_variance_ratio_  # calculate variance ratios
    var = np.cumsum(np.round(variance, decimals=5) * 100)
    print(var)  # cumulative sum of variance explained with [n] features
    plot.plot_pca_variance(var)


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


if __name__ == '__main__':
    # number of nearest neighbors to check

    # load training and test images
    training_lists = load_image_vectors.load_gz('../data/mnist_train.csv.gz')
    print("Successfully loaded training list")
    test_lists = load_image_vectors.load_gz('../data/mnist_test.csv.gz')
    print("Successfully loaded test list")

    sklearn_k_value_test(1, 5)
    pca_variance_analysis(test_lists[1])
