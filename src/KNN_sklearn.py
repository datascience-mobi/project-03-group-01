import sklearn
from sklearn import neighbors, metrics
import src.pickle_operations as pickle_io


"the function takes two tuples: train and test; N_neighbours for KNN; M_labels for output "
"it reads the labels of test images and creates the 784 dim. space,"
"in which the points from test_tuple will be entered."
" then the KNN using N neighbours  with the points from test_tuple will be performed"
"output - M_labels of test_tuple"


def knn_sk(test_images, train_images, n_neighbours, min_index, max_index):
    """
    performs the sklearn knn implementation from test image number min_index up to max_index -1
    :param test_images: list of test images as CsvImages
    :param train_images:  list of training images as CsvImages
    :param n_neighbours: number of neighbors for KNN
    :param min_index: lowest test image number to perform the knn for
    :param max_index: lowest test image number NOT to perform the knn for
    :return: 2d list of type [[10, 0], [11, 6]] meaning [test image number, prediction]
    """
    test_pred = list()
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbours).fit([csv_image.image for csv_image in train_images], [csv_image.label for csv_image in train_images])
    for j in range(min_index, max_index):
        test_pred.append([csv_image.image for csv_image in test_images][j])
        # print(j)
    print("finished that loop")
    test_prediction = knn.predict(test_pred).tolist()
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
        print(f"inside knn: {i}")
        test_prediction = knn.predict_proba([csv_image.image for csv_image in test_images if csv_image.label == i]).tolist()
        print(test_prediction)
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
            print(f"success: i = {i}")
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
    print(f"max_accuracy: {max_accuracy}, max_label: {max_label}, true label: {best_index}")

    return best_index


# if __name__ == '__main__':
#     training_lists = pickle_io.load_pickles('../data/training.dat')
#     print("Successfully loaded training list")
#     test_lists = pickle_io.load_pickles('../data/test.dat')
#     print("Successfully loaded test list")
#
#     # get list of a list of all probabilities that a certain image displays a certain digit
#     all_predictions = (knn_sk_probabilities(test_lists, training_lists, 500))
#     pickle_io.save_pickles(all_predictions, "../data/skknnproba.dat")
#     # --- run code above once to create the .dat then only run line below ----
#     # all_predictions = pickle_io.load_pickles("../data/skknnproba.dat")
#
#     # get labels of most clearly recognized images for each digit
#     best_images = list()
#     for i in range(10):
#         print(f"i: {i}")
#         best_images.append(get_most_unique_image(all_predictions[i], i, test_lists))
#     print(f"Best images: {best_images}")
#     # save calculated labels for the best / most clearly recognized test images in a txt file
#     with open('best_digits.txt', 'w') as the_file:
#         for index in best_images:
#             the_file.write(f"{index}\n")