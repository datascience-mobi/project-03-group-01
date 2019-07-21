import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import src.KNN_sklearn as KNN_sklearn
import src.pickle_operations as pickle_io


def get_best_digits(training_lists, test_lists):
    """
    Calculates most unique / less misrecognized images for each digit
    :param training_lists: training images
    :param test_lists: test images
    :return: best images
    """

    # # get list of a list of all probabilities that a certain image displays a certain digit
    # all_predictions = (KNN_sklearn.knn_sk_probabilities(test_lists, training_lists, 500))
    # pickle_io.save_pickles(all_predictions, "data/skknnproba.dat")
    # # --- run code above once to create the .dat then only run line below ----
    all_predictions = pickle_io.load_pickles("data/skknnproba.dat")

    # get labels of most clearly recognized images for each digit
    best_images = list()
    for i in range(10):
        # Calls function that finds image that's mostly surrounded by images of the same digit
        best_images.append(test_lists[KNN_sklearn.get_most_unique_image(all_predictions[i], i, test_lists)])
    return best_images


def get_mean_digits(test_images):
    """
    Gets mean value of all test image pixels
    :param test_images: test images
    :return: mean digits as list
    """
    mean_digits = list()
    for i in range(10):
        training_array = np.asarray([csv_image.image for csv_image in test_images if csv_image.label == i])
        mean_digits.append(training_array.mean(axis=0))
    return mean_digits


def get_median_digits(test_images):
    """
    calculates median of all test image pixels accordingly for each digit
    :param test_images: test images
    :return: median digits as list
    """
    median_digits = list()
    for i in range(10):
        training_array = np.asarray([csv_image.image for csv_image in test_images if csv_image.label == i])
        median_digits.append(np.median(training_array,axis=0))
    return median_digits


def show_best_digits(best_digits):
    """
    shows best digits as plot
    :param best_digits: best digits
    :return: None
    """
    plt.figure(figsize=(5, 10))
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
    plt.tight_layout()
    plt.show()


def show_mean_digits(test_images):
    """
    gets image list and calls function to calculate mean digits, then plots them
    :param test_images: test images
    :return: None
    """
    mean_digits = get_mean_digits(test_images)
    plt.figure(figsize=(5, 10))
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
    plt.tight_layout()
    plt.show()


def show_median_digits(test_images):
    """
    gets image list and calls function to calculate mean digits, then plots them
    :param test_images: test images
    :return: None
    """
    median_digits = get_median_digits(test_images)
    plt.figure(figsize=(5, 10))
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
    plt.tight_layout()
    plt.show()