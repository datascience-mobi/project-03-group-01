import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, metrics
import matplotlib.pyplot as plt
import numpy as np
from src import pca
from src import KNN_sklearn


def show_wrong_predicted(train_images, test_images, best_digits):
    """
    Plots 5 falsely recognized 7's next to the optimal 7 and the optimal
    :param train_images: training images
    :param test_images: list of test images
    :param best_digits: Most uniquely recognized images (meta digits)
    :return: None
    """
    # Runs PCA
    reduced_images = pca.reduce_dimensions([csv_image.image for csv_image in train_images], [csv_image.image for csv_image in test_images], 78)

    # Runs KNN for all test images
    test_predictions = KNN_sklearn.knn_sk(reduced_images[0], reduced_images[1], [csv_image.label for csv_image in train_images], 3, 0, 10000)

    # number of false recognized images already plotted
    count = 0

    plt.figure(figsize=(5, 10))
    for i in range(len(test_predictions)):
        if test_images[test_predictions[i][0]].label != test_predictions[i][1] and test_images[test_predictions[i][0]].label == 7:
            plt.subplot(5, 3, 3 * count + 1)
            plt.imshow(np.asarray(best_digits[test_images[test_predictions[i][0]].label].reshape(28, 28)),
                       cmap=plt.cm.gray, interpolation='nearest',
                       clim=(0, 255))
            plt.xlabel(f"Best recognized {test_images[test_predictions[i][0]].label}", fontsize=14)
            plt.subplot(5, 3, 3 * count + 2)
            plt.imshow(np.asarray(test_images[test_predictions[i][0]].image).reshape(28, 28),
                       cmap=plt.cm.gray, interpolation='nearest',
                       clim=(0, 255))
            plt.xlabel(f"prediction: {test_predictions[i][1]}", fontsize=14)
            plt.subplot(5, 3, 3 * count + 3)
            plt.imshow(np.asarray(best_digits[test_predictions[i][1]].reshape(28, 28)),
                       cmap=plt.cm.gray, interpolation='nearest',
                       clim=(0, 255))
            plt.xlabel(f"Best recognized {test_predictions[i][1]}", fontsize=14)
            count += 1
            if count > 4:
                break
    plt.tight_layout()
    plt.show()


def plot_sample_recognitions(training_lists, test_lists, k):
    """
    Plots first 10 test images + their prediction
    :param training_lists: training images
    :param test_lists: test images
    :param k: for which k to run the PCA
    :return: None
    """
    pred = knn_sk([csv_image.image for csv_image in training_lists], [csv_image.image for csv_image in test_lists], [csv_image.label for csv_image in training_lists], k, 0, 10)
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(np.asarray(test_lists[i].image).reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest',
                   clim=(0, 255))
        plt.xlabel(f"prediction: {pred[i][1]}", fontsize=14)
    plt.tight_layout()
    plt.show()


def knn_sk(train_images, test_images, train_labels, n_neighbours, min_index, max_index):
    """
    performs the sklearn knn implementation from test image number min_index up to max_index -1
    :param test_images: list of test images as CsvImages
    :param train_images:  list of training images as CsvImages
    :param train_labels: list of labels of train images
    :param n_neighbours: number of neighbors for KNN
    :param min_index: lowest test image number to perform the knn for
    :param max_index: lowest test image number NOT to perform the knn for
    :return: 2d list of type [[10, 0], [11, 6]] meaning [test image number, prediction]
    """

    test_pred = list()
    # define knn function using train data as a space for prediction
    # the function needs two lists: of data points in 784 dim. space and of their labels
    # they are written in addition .fit ()
    # n_neighbours defines how many neighbours we want to take while making prediction
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbours).fit(train_images, train_labels)
    # loop for prediction of only specific wanted digits between min_index and max_index-1
    # the actual points of test data set are extracted as test_pred
    for j in range(min_index, max_index):
        test_pred.append(test_images[j])
    # application of knn to the extracted test points and converting to list
    test_prediction = knn.predict(test_pred).tolist()
    # creating 2D list by pairing the elements of two tuples: range of wanted indexes and test_prediction
    # 2d list has this form [index/position of the digit1][prediction made by knn],... for every digits in a given range
    # this list is our output
    test_prediction = [list(a) for a in zip([j for j in range(min_index, max_index)], test_prediction)]
    return test_prediction


def knn_sk_probabilities(test_images, train_images, n_neighbours):
    """
    creates list of probabilities which digit an image belongs to for each image and digit
    :param test_images: list of test images as CsvImages
    :param train_images:  list of training images as CsvImages
    :param n_neighbours: number of neighbors for KNN
    :return: 1d list containing 10 times 2d list of type [[10, [1.0,0.0...0.0], [11, 0.0,1.0...0.0]]
        meaning [test image number, [probabilities for each digit]
    """
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbours).fit([csv_image.image for csv_image in train_images], [csv_image.label for csv_image in train_images])
    print("Finished KNeighborsClassifier") # takes most time

    # create a list of probabilities for each digit and the corresponding images
    test_predictions = list()
    for i in range(10):
        test_prediction = knn.predict_proba([csv_image.image for csv_image in test_images if csv_image.label == i]).tolist()
        # assigns the corresponding label to each prediction
        test_prediction = [list(a) for a in zip([j for j in range(len(test_images))], test_prediction)]
        print(test_prediction)
        test_predictions.append(test_prediction)
    return test_predictions


def get_total_index(image_list, digit, local_index) -> int:
    """
    translates the index of an image within all images of a certain digit to an index within all images of a list
    :param image_list: list of images of all sorts of digits
    :param digit: what digit does the local_index belong to
    :param local_index: index for an image within the sublist of a certain type of digits within a list
    :return: index within all images, not just those of a certain digit
    """
    j = -1
    for i, csv_image in enumerate(image_list):
        if csv_image.label == digit:
            # j counts how many images of a certain label were already found
            j += 1
        if j == local_index:
            # directly end the loop as soon as the total index was found
            return i


def get_most_unique_image(predictions, label, test_images) -> int:
    """
    determines image that got best recognized as what digit it is
    :param predictions: list of probabilities to belong to a certain digit for all images of a certain digit
    :param label: what digits are described in predictions
    :param test_images: list of all test images, only relevant for later calling get_total_index
    :return: index of the most unique image in the original test_list
    """
    # image with best accuracy + its accuracy
    max_accuracy = 0.0
    max_label = -1

    for idx, pred in enumerate(predictions):
        # is the probability that this image is what it is higher than the maximum one?
        if pred[1][label] > max_accuracy:
            max_accuracy = pred[1][label]
            max_label = idx

    # max_label is the index within all images of the same label,
    # thus the index within test_images needs to be determined
    best_index = get_total_index(test_images, label, max_label)

    return best_index
