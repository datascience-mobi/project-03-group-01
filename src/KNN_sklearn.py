import sklearn
import src.load_image_vectors as load_image_vectors
import src.pickle_operations as pickle_io
from sklearn import neighbors, metrics
import src.knn as knn



"the function takes two tuples: train and test; N_neighbours for KNN; M_labels for output "
"it reads the labels of test images and creates the 784 dim. space,"
"in which the points from test_tuple will be entered."
" then the KNN using N nighbours  with the points from test_tuple will be performed"
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
    # test_pred = list()
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbours).fit([csv_image.image for csv_image in train_images], [csv_image.label for csv_image in train_images])
    print("Finished KNeighborsClassifier") # takes most time
    # for j in range(len(test_images)):
    #     test_pred.append([csv_image.image for csv_image in test_images][j])
    #     # print(j)
    # print(test_pred == test_images)
    test_predictions = list()
    for i in range(10):
        print(f"inside knn: {i}")

        test_prediction = knn.predict_proba([csv_image.image for csv_image in test_images if csv_image.label == i]).tolist()
        # print(knn.predict_proba(test_pred))
        test_prediction = [list(a) for a in zip([j for j in range(len(test_images))], test_prediction)]
        print(test_prediction)
        test_predictions.append(test_prediction)
    return test_predictions


def get_total_index(image_list, digit, local_index):
    j = -1
    for i, img in enumerate(image_list):
        # print(f"{i}, {img.label}")
        if img.label == digit:
            j += 1
        if j == local_index:
            print(f"success: i = {i}")
            return i


def get_most_unique_image(preds, label):
    """
    determines image that got best recognized as what digit it is
    :param preds: list of probabilities
    :param label:
    :return:
    """
    max_accuracy = 0.0
    max_label = -1
    for idx, pred in enumerate(preds):
        # print(pred[1][5])
        if pred[1][label] > max_accuracy:
            max_accuracy = pred[1][label]
            max_label = idx
    true_label = get_total_index(test_lists, label, max_label)
    print(f"max_accuracy: {max_accuracy}, max_label: {max_label}, true label: {true_label}")
    return true_label


"for test purposes"
if __name__ == '__main__':
    training_lists = pickle_io.load_pickles('../data/training.dat')
    print("Successfully loaded training list")
    test_lists = pickle_io.load_pickles('../data/test.dat')
    print("Successfully loaded test list")
    # print(training_lists[get_total_index(5, 623)].image)
    # print([train for train in training_lists if train.label == 5][624].image)
    # print(training_lists[get_total_index(5, 623)].image == [train for train in training_lists if train.label == 5][623].image)
    best_images = list()
    all_predictions = (knn_sk(test_lists, training_lists, 500, 0, 5999))
    pickle_io.save_pickles(all_predictions, "../data/skknnproba.dat")
    for i in range(10):
        print(f"i: {i}")
        # predictions = (knn_sk([test for test in test_lists if test.label == i], training_lists, 500, 0, 5999))
        best_images.append(get_most_unique_image(all_predictions[i], i))
    print(f"Best images: {best_images}")
    with open('best_digits.txt', 'w') as the_file:
        for index in best_images:
            the_file.write('index')