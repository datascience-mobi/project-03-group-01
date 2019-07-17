import matplotlib.pyplot as plt
import numpy as np
import math
from src import pickle_operations as pickle_io
from src import knn_sklearn_copy as knn_sklearn


def get_best_digits(training_lists, test_lists):

    # # get list of a list of all probabilities that a certain image displays a certain digit
    # all_predictions = (knn_sklearn.knn_sk_probabilities(test_lists, training_lists, 500))
    # pickle_io.save_pickles(all_predictions, "../data/skknnproba.dat")
    # # # --- run code above once to create the .dat then only run line below ----
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

    # plt.subplot(3, 2, 3)
    # plt.imshow(mean_digit.reshape(28, 28),
    #            cmap=plt.cm.gray, interpolation='nearest',
    #            clim=(0, 255))
    # plt.xlabel('Mean of all training '+str(digit)+"s", fontsize=14)
    # plt.subplot(3, 2, 4)
    # plt.imshow(median_digit.reshape(28, 28),
    #            cmap=plt.cm.gray, interpolation='nearest',
    #            clim=(0, 255))
    # plt.xlabel('Median of all training '+str(digit)+"s", fontsize=14)
    # plt.subplot(3, 2, 5)
    # plt.imshow(test_image.reshape(28, 28),
    #            cmap=plt.cm.gray, interpolation='nearest',
    #            clim=(0, 255))
    # plt.imshow(mean_digit.reshape(28, 28),
    #            cmap=plt.cm.gray, interpolation='nearest',
    #            clim=(0, 255), alpha=0.5)
    #
    # plt.xlabel('Your drawn image + Mean', fontsize=14)
    # plt.subplot(3, 2, 6)
    # plt.imshow(test_image.reshape(28, 28),
    #            cmap=plt.cm.gray, interpolation='nearest',
    #            clim=(0, 255))
    # plt.imshow(median_digit.reshape(28, 28),
    #            cmap=plt.cm.gray, interpolation='nearest',
    #            clim=(0, 255), alpha=0.5)

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


if __name__ == '__main__':
    plot_grouped_distances([20, 35, 30, 35, 27], [25, 32, 34, 20, 25], False)