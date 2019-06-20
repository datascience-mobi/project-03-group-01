import matplotlib.pyplot as plt
import numpy as np


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
        median_digits.append(np.median(training_array,axis=0))
    return median_digits


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