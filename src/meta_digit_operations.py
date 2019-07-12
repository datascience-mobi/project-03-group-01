import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import src.KNN_sklearn as KNN_sklearn
import src.pickle_operations as pickle_io


def show_as_heatmap(mean_digits, median_digits):
    heat_list_mean = list()
    plt.subplot(1, 2, 1)
    # min_dist_mean = 0
    for i in range(len(mean_digits)):
        for j in range(len(mean_digits)):
            dist = distance.euclidean(mean_digits[i], mean_digits[j])
            # if dist < min_dist_mean and not dist == 0:
            #     min_dist_mean = dist
            heat_list_mean.append(dist)
    min_dist_mean = min([heat for heat in heat_list_mean if not heat == 0])
    max_dist_mean = max(heat_list_mean)
    # a = np.random.random((16, 16))
    heat_list_mean = np.asarray(heat_list_mean)

    plt.imshow(heat_list_mean.reshape(10, 10), cmap='hot', interpolation='nearest')
    plt.xlabel(f"Distances among Mean Digits \n Min: {min_dist_mean} \n Max: {max_dist_mean} \n Diff: {min_dist_mean/max_dist_mean}")
    plt.subplot(1, 2, 2)
    heat_list_median = list()
    min_dist_median = 0
    for i in range(len(median_digits)):
        for j in range(len(mean_digits)):
            dist = distance.euclidean(median_digits[i], median_digits[j])
            if dist < min_dist_median and not dist == 0:
                min_dist_median = dist
            heat_list_median.append(dist)
    # a = np.random.random((16, 16))
    min_dist_median = min([heat for heat in heat_list_median if not heat == 0])
    max_dist_median = max(heat_list_median)

    heat_list = np.asarray(heat_list_median)

    plt.imshow(heat_list.reshape(10,10), cmap='hot', interpolation='nearest')
    plt.xlabel(f"Distances among Median Digits \n Min: {min_dist_median} \n Max: {max_dist_median} \n Diff: {min_dist_median/max_dist_median}")

    plt.show()


def get_best_digits(training_lists, test_lists):

    # # get list of a list of all probabilities that a certain image displays a certain digit
    # all_predictions = (KNN_sklearn.knn_sk_probabilities(test_lists, training_lists, 500))
    # pickle_io.save_pickles(all_predictions, "../data/skknnproba.dat")
    # # --- run code above once to create the .dat then only run line below ----
    all_predictions = pickle_io.load_pickles("../data/skknnproba.dat")

    # get labels of most clearly recognized images for each digit
    best_images = list()
    for i in range(10):
        print(f"i: {i}")
        best_images.append(test_lists[KNN_sklearn.get_most_unique_image(all_predictions[i], i, test_lists)])
    return best_images


def get_mean_digits(training_images):
    mean_digits = list()
    for i in range(10):
        training_array = np.asarray([csv_image.image for csv_image in training_images if csv_image.label == i])
        mean_digits.append(training_array.mean(axis=0))
    return mean_digits


def mean_center_distance(training_images, center):
    center_distance = list()
    for train in training_images:
        center_distance.append(np.linalg.norm(train - center))
    return np.mean(center_distance)


def get_median_digits(training_images):
    median_digits = list()
    for i in range(10):
        training_array = np.asarray([csv_image.image for csv_image in training_images if csv_image.label == i])
        median_digits.append(np.median(training_array,axis=0))
    return median_digits


def show_best_digits(training_images, test_images):
    best_digits = np.asarray([csv_image.image for csv_image in get_best_digits(training_images, test_images)])
    for i in range(5):
        # left image - even digit
        plt.subplot(5, 2, 2*i+1)
        plt.imshow(best_digits[2*i].reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest',
                   clim=(0, 255))
        plt.xlabel('Most clearly recognized '+str(2*i), fontsize=14)

        # right image - odd digit
        plt.subplot(5, 2, 2*i+2)
        plt.imshow(best_digits[2*i+1].reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest',
                   clim=(0, 255))
        plt.xlabel('Most clearly recognized '+str(2*i+1), fontsize=12)
    plt.show()


def show_mean_digits(training_images):
    mean_digits = get_mean_digits(training_images)
    for i in range(5):
        # left image - even digit
        plt.subplot(5, 2, 2*i+1)
        plt.imshow(mean_digits[2*i].reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest',
                   clim=(0, 255))
        plt.xlabel('Mean of all '+str(2*i)+"'s", fontsize=14)

        # right image - odd digit
        plt.subplot(5, 2, 2*i+2)
        plt.imshow(mean_digits[2*i+1].reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest',
                   clim=(0, 255))
        plt.xlabel('Mean of all '+str(2*i+1)+"'s", fontsize=12)
    plt.show()


def show_median_digits(training_images):
    median_digits = get_median_digits(training_images)
    for i in range(5):
        plt.subplot(5, 2, 2*i+1)
        plt.imshow(median_digits[2*i].reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest',
                   clim=(0, 255))
        plt.xlabel('Median of all '+str(2*i)+"'s", fontsize=14)

        plt.subplot(5, 2, 2*i+2)
        plt.imshow(median_digits[2*i+1].reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest',
                   clim=(0, 255))
        plt.xlabel('Median of all '+str(2*i+1)+"'s", fontsize=12)
    plt.show()