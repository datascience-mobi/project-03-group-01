import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import src.pca as pca
import src.KNN_sklearn as KNN_sklearn
import src.pickle_operations as pickle


def k_accuracy_test(train_list, test_list, k_min, k_max):
    """
    runs the knn-algorithm for different k-values
    creates a list in the form of [[k1, accuracy1], [k2, accuracy2], ...]
    then saves the list
    :param train_list: training images
    :param test_list: test images
    :param k_min: smallest k value
    :param k_max: biggest k value
    :return: list in the form of [[k1, accuracy1], [k2, accuracy2], ...]
    """

    k_accuracy = list()
    for k in range(k_min, k_max):
        success_number = 0
        total_number = 0
        prediction_list = KNN_sklearn.knn_sk([csv_image.image for csv_image in train_list], [csv_image.image for csv_image in test_list], [csv_image.label for csv_image in train_list], k, 1, 10000)
        for idx, prediction in enumerate(prediction_list):
            total_number += 1
            if prediction[1] == test_list[prediction[0]].label:  # counts the number of correct predictions
                success_number += 1
        print(f"total number:{total_number}, success number:{success_number}")
        accuracy = float(success_number) / float(total_number)  # calculates accuracy
        k_accuracy.append([k, accuracy])
        print("Finished accuracy calculation " + str(k))
    print('k_accuracy = ', k_accuracy)
    pickle.save_pickles(k_accuracy, "k_accuracy2.dat")  # saves the list because it takes a lot of time
    return k_accuracy


def pca_variance_analysis(input_list):
    """
    creates covariance matrix of training images and gets explained variance for each dimension -> plots them
    :param input_list: training images
    :return: None
    """
    # calculates the retained variance for different dimensions and plots the resulting variance-matrix
    # print("Started pca_variance_analysis")
    scale = preprocessing.StandardScaler()
    scale.fit(input_list)
    input_list = scale.transform(input_list)
    covar_matrix = PCA(n_components=784)  # we have 784 features
    covar_matrix.fit(input_list)
    variance = covar_matrix.explained_variance_ratio_  # calculate variance ratios
    var = np.cumsum(np.round(variance, decimals=5) * 100)  # cumulative sum of variance explained with [n] features

    # plot yielded variance
    plot_pca_variance(var)


def pca_accuracy_test(test_lists, training_lists, steps):
    """
    Runs PCA for different target number of dimensions, then runs KNN wiht k=3 to get the accuracy for that dimension
    :param test_lists: test images
    :param training_lists: training images
    :param steps: determines number of dimensions to test for
    :return: accuracy for each dimension
    """
    # runs the knn-algorithm (k=3) for different pca-parameters
    # then saves the accuracy in a list formed like [[n1, accuracy1], [n2, accuracy2], ... ]

    print("Started pca_accuracy_test")
    pca_accuracy = list()
    n_steps = list()
    if steps == 1:
        n_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
                120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 700, 784]
    else:
        n_steps = [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86] # for testing purposes
    # those dimensions were chosen for the pca

    for n in n_steps:
        print(f"n = {n}")
        reduced_images = pca.reduce_dimensions([csv_image.image for csv_image in training_lists], [csv_image.image for csv_image in test_lists], n)
        all_images = 0
        success_images = 0
        prediction_list = KNN_sklearn.knn_sk(reduced_images[0], reduced_images[1], [csv_image.label for csv_image in training_lists], 3, 1, 10000)
        for idx, prediction in enumerate(prediction_list):
            all_images += 1
            if prediction[1] == test_lists[prediction[0]].label:  # this counts the correctly recognized images
                success_images += 1
        accuracy = float(success_images) / float(all_images)  # calculates accuracy
        pca_accuracy.append([n, accuracy])
        print("Finished accuracy calculation " + str(n))
    print('pca_accuracy = ', pca_accuracy)
    pickle.save_pickles(pca_accuracy, f"pca_accuracy_gen{steps}.dat")  # saves the list because it takes time
    return pca_accuracy


def plot_pca_accuracy(input_list):
    """
    plots accuracies for different dimensions as bar chart
    :param input_list: accuracies of different dimensions
    :return: None
    """
    # plots a list in the form of [[n1, accuracy], [n2, accuracy], ...] as a barplot

    labels, ys = zip(*input_list)
    xs = np.arange(len(labels))
    width = 0.8
    plt.bar(xs, ys, width, align='center')
    frequency = 3  # only every third bar gets a label so they dont overlap
    plt.xticks(xs[::frequency], labels[::frequency])

    plt.ylabel('Accuracy')
    plt.xlabel('#n')
    plt.title('Accuracy test')
    plt.show()


def plot_k_accuracy(input_list):
    """
    Plots accuracy for different k's
    :param input_list: accuracy for different k values
    :return: None
    """
    # plots a list in the form of [[k1, accuracy], [k2, accuracy], ...] as a barplot

    labels, ys = zip(*input_list)
    xs = np.arange(len(labels))
    width = 0.8
    plt.bar(xs, ys, width, align='center')
    plt.xticks(xs, labels)  # x-axis labeling

    plt.ylabel('Accuracy')
    plt.xlabel('k')
    plt.title('Accuracy of various k values')
    plt.ylim(0.96, 0.975)  # limit y axis to see differences
    plt.show()


def plot_pca_variance(input_list):
    """
    plots a the covariance matrix as a graph
    :param input_list: explained variance of principal components of training images with n=784
    :return: None
    """

    plt.ylabel('% Variance Explained')
    plt.xlabel('Dimensions')
    plt.title('PCA analysis')
    plt.ylim(10, 100.5)  # limits y-axis
    plt.xlim(-10, 784)  # limits x-axis
    plt.plot(input_list)
    plt.show()
