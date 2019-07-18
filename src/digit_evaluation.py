import matplotlib.pyplot as plt
import numpy as np
import math
from src import pickle_operations as pickle_io
from src import knn_sklearn_copy as knn_sklearn


def get_best_digits(training_lists, test_lists):

    # # get list of a list of all probabilities that a certain image displays a certain digit
    all_predictions = (knn_sklearn.knn_sk_probabilities(test_lists, training_lists, 500))
    pickle_io.save_pickles(all_predictions, "../data/skknnproba.dat")
    # --- run code above once to create the .dat then only run line below ----
    all_predictions = pickle_io.load_pickles("../data/skknnproba.dat")

    # get labels of most clearly recognized images for each digit
    best_images = list()
    for i in range(10):
        best_images.append(test_lists[knn_sklearn.get_most_unique_image(all_predictions[i], i, test_lists)])
    return best_images


def show_difference(test_image, digit, best_digit, evaluation):
    plt.figure().suptitle(f"Score: {evaluation}", fontsize="x-large")
    plt.subplot(1, 2, 1)
    plt.imshow(test_image.reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 255))
    plt.xlabel('Your drawn image', fontsize=14)
    plt.subplot(1, 2, 2)
    plt.imshow(best_digit.reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 255))
    plt.xlabel(f"Best recognizable {digit}", fontsize=14)

    plt.show()


def plot_grouped_distances(mean_distances, median_distances, zoomed):

    below_threshold = min(mean_distances + median_distances)

    ind = np.arange(len(mean_distances))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(ind - width / 2, mean_distances, width,
           label='Mean digits')

    ax.bar(ind + width / 2, median_distances, width,
           label='Median Digits')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Distances')
    ax.set_xlabel('Digits')
    ax.set_title('Distance between drawn digit and meta digits')
    ax.set_xticks(ind)
    ax.set_xticklabels(get_x_ticks_label())

    if zoomed:
        axes = plt.gca()
        axes.set_ylim([below_threshold-math.sqrt(below_threshold), None])

    ax.legend()
    ax.plot([-0.5, 9.5], [below_threshold, below_threshold], "k--")
    fig.tight_layout()

    plt.show()


def get_x_ticks_label():
    labels = list()
    for i in range(10):
        labels.append(i)
    return labels


def get_mean_digits(training_images):
    mean_digits = list()
    for i in range(10):
        training_array = np.asarray([csv_image.image for csv_image in training_images if csv_image.label == i])
        mean_digits.append(training_array.mean(axis=0))
    return mean_digits


def get_median_digits(training_images):
    median_digits = list()
    for i in range(10):
        training_array = np.asarray([csv_image.image for csv_image in training_images if csv_image.label == i])
        median_digits.append(np.median(training_array, axis=0))
    return median_digits


def show_mean_digits(training_images):
    mean_digits = get_mean_digits(training_images)
    for i in range(5):
        # left image - even digit
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(mean_digits[2 * i].reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest',
                   clim=(0, 255))
        plt.xlabel('Mean of all ' + str(2 * i) + "'s", fontsize=14)

        # right image - odd digit
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(mean_digits[2 * i + 1].reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest',
                   clim=(0, 255))
        plt.xlabel('Mean of all ' + str(2 * i + 1) + "'s", fontsize=12)
    plt.show()


def show_median_digits(training_images):
    median_digits = get_median_digits(training_images)
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(median_digits[2 * i].reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest',
                   clim=(0, 255))
        plt.xlabel('Median of all ' + str(2 * i) + "'s", fontsize=14)

        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(median_digits[2 * i + 1].reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest',
                   clim=(0, 255))
        plt.xlabel('Median of all ' + str(2 * i + 1) + "'s", fontsize=12)
    plt.show()

if __name__ == '__main__':
    plot_grouped_distances([20, 35, 30, 35, 27], [25, 32, 34, 20, 25], False)