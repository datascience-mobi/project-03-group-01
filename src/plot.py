import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import src.pca as pca
import src.KNN_sklearn as KNN_sklearn
import src.pickle_operations as pickle


def k_accuracy_test(train_list, test_list, k_min, k_max):
    # runs the knn-algorithm for different k-values
    # creates a list in the form of [[k1, accuracy1], [k2, accuracy2], ...]
    # then saves the list

    k_accuracy = list()
    print("Started sklearn_k_value_test")
    for k in range(k_min, k_max):
        success_number = 0
        total_number = 0
        prediction_list = KNN_sklearn.knn_sk([csv_image.image for csv_image in train_list], [csv_image.image for csv_image in test_list], [csv_image.label for csv_image in train_list], k, 1, 5)
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
    # calculates the retained variance for different dimensions and plots the resulting variance-matrix
    # print("Started pca_variance_analysis")
    scale = preprocessing.StandardScaler()
    scale.fit(input_list)
    input_list = scale.transform(input_list)
    covar_matrix = PCA(n_components=784)  # we have 784 features
    covar_matrix.fit(input_list)
    variance = covar_matrix.explained_variance_ratio_  # calculate variance ratios
    var = np.cumsum(np.round(variance, decimals=5) * 100)
    # print(var)  # cumulative sum of variance explained with [n] features
    plot_pca_variance(var)


def pca_accuracy_test(test_lists, training_lists, steps):
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

#
# def save_results():
#     plt.savefig("accuracy_test.png")  # save the figure to file as png
#     plt.savefig("accuracy_test.pdf")  # save the figure to file as pdf


def plot_pca_accuracy(input_list):
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
    # plots a the covariance matrix as a graph

    plt.ylabel('% Variance Explained')
    plt.xlabel('Dimensions')
    plt.title('PCA analysis')
    plt.ylim(10, 100.5)  # limits y-axis
    plt.xlim(-10, 784)  # limits x-axis
    plt.plot(input_list)
    plt.show()


def perfect_k_for_perfect_n(k_min, k_max, test_lists, training_lists):
    perfect_k2 = list()
    reduced_images = pca.reduce_dimensions([csv_image.image for csv_image in training_lists],
                                           [csv_image.image for csv_image in test_lists], 78)
    for k in range(k_min, k_max):
        print(f"k = {k}")
        success_number = 0
        total_number = 0
        prediction_list = KNN_sklearn.knn_sk(reduced_images[0], reduced_images[1],
                                             [csv_image.label for csv_image in training_lists], k, 1, 10000)
        for idx, prediction in enumerate(prediction_list):
            total_number += 1
            if prediction[1] == test_lists[prediction[0]].label:  # counts the number of correct predictions
                success_number += 1
        print(f"total number:{total_number}, success number:{success_number}")
        accuracy = float(success_number) / float(total_number)  # calculates accuracy
        perfect_k2.append([k, accuracy])
    pickle.save_pickles(perfect_k2, "perfect_k2.dat")  # saves the list because it takes time
    return perfect_k2


# if __name__ == '__main__':
#     '''Whatever you do,DO NOT change k_accuracy.dat under any circumstances'''
#     # for testing purposes
#
#     # load training and test images
#     training_lists = pickle.load_pickles("../data/training.dat")
#     test_lists = pickle.load_pickles("../data/test.dat")
#     print("Successfully loaded images from pickle files")
#     # k_accuracy_test(1, 4)
#     # plot_k_accuracy(pickle.load_pickles("k_accuracy.dat"))
#     # pca_variance_analysis([csv_image.image for csv_image in test_lists])
#     # pca_accuracy_test(test_lists, training_lists)
#     # plot_pca_accuracy(pickle.load_pickles("pca_accuracy.dat"))
#     # print(pickle.load_pickles("pca_accuracy.dat"))
#     # perfect_k_for_perfect_n(2, 6, test_lists, training_lists)
#     # plot_k_accuracy(pickle.load_pickles("perfect_k2.dat"))

