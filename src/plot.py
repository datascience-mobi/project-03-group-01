import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import src.pca as pca

import src.KNN_sklearn as KNN_sklearn
import src.load_image_vectors as load_image_vectors
import src.pickle_operations as pickle
from itertools import accumulate


def k_accuracy_test(k_min, k_max):
    # creates a list in the form of [[k1, accuracy1], [k2, accuracy2], ...]
    # then saves the list
    # then plots the list
    k_accuracy = list()
    print("Started sklearn_k_value_test")
    for k in range(k_min, k_max):
        success_number = 0
        total_number = 0
        prediction_list = KNN_sklearn.knn_sk([csv_image.image for csv_image in training_lists], [csv_image.image for csv_image in test_lists], [csv_image.label for csv_image in training_lists], k, 1, 5)
        for idx, prediction in enumerate(prediction_list):
            total_number += 1
            if prediction[1] == test_lists[prediction[0]].label:
                success_number += 1
        print(f"total number:{total_number}, success number:{success_number}")
        accuracy = float(success_number) / float(total_number)
        k_accuracy.append([k, accuracy])
        print("Finished accuracy calculation " + str(k))
    print('k_accuracy = ', k_accuracy)
    pickle.save_pickles(k_accuracy, "k_accuracy2.dat")
    # plot_accuracy(pickle.load_pickles("k_accuracy2.dat")) for testing purposes
    return k_accuracy


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


def pca_accuracy_test(test_lists, training_lists):
    print("Started pca_accuracy_test")
    pca_accuracy = list()
    n_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
                120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 700, 784]
    for n in n_steps:
        print(f"n = {n}")
        reduced_images = pca.reduce_dimensions([csv_image.image for csv_image in training_lists], [csv_image.image for csv_image in test_lists], n)
        all_images = 0
        success_images = 0
        prediction_list = KNN_sklearn.knn_sk(reduced_images[0], reduced_images[1], [csv_image.label for csv_image in training_lists], 3, 1, 10000)
        for idx, prediction in enumerate(prediction_list):
            all_images += 1
            if prediction[1] == test_lists[prediction[0]].label:
                success_images += 1
        accuracy = float(success_images) / float(all_images)
        pca_accuracy.append([n, accuracy])
        print("Finished accuracy calculation " + str(n))
    print('pca_accuracy = ', pca_accuracy)
    pickle.save_pickles(pca_accuracy, "pca_accuracy2.dat")
    return pca_accuracy


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
    plt.savefig("accuracy_test.png")  # save the figure to file as png
    plt.savefig("accuracy_test.pdf")  # save the figure to file as pdf


def plot_accuracy(input_list):
    # plots a list in the form of [[k1, accuracy], [k2, accuracy], ...] as a diagram

    labels, ys = zip(*input_list)
    xs = np.arange(len(labels))
    width = 0.8
    plt.bar(xs, ys, width, align='center')
    plt.xticks(xs, labels)
    plt.yticks(ys)

    plt.ylabel('Accuracy')
    plt.xlabel('#n')
    plt.title('Accuracy test')
    plt.ylim(0.9, 0.972)  # limit y axis to see differences
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
    '''Whatever you do,DO NOT change k_accuracy.dat under any circumstances'''
    # for testing purposes

    # load training and test images
    # training_lists = pickle.load_pickles("../data/training.dat")
    # test_lists = pickle.load_pickles("../data/test.dat")
    # print("Successfully loaded images from pickle files")

    # k_accuracy_test(1, 4)
    # plot_accuracy(pickle.load_pickles("k_accuracy2.dat"))
    # pca_variance_analysis(test_lists[1])
    # pca_accuracy_test(test_lists, training_lists)
    plot_accuracy(pickle.load_pickles("pca_accuracy.dat"))
    print(pickle.load_pickles("pca_accuracy.dat"))
