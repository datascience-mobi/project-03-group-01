import sklearn
import src.load_image_vectors as load_image_vectors
import src.pickle_operations as pickle_io
from sklearn import neighbors, metrics


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
    test_pred = list()
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbours).fit([csv_image.image for csv_image in train_images], [csv_image.label for csv_image in train_images])
    for j in range(min_index, max_index):
        test_pred.append([csv_image.image for csv_image in test_images][j])
        print(j)
    test_prediction = knn.predict(test_pred).tolist()
    test_prediction = [list(a) for a in zip([j for j in range(min_index, max_index)], test_prediction)]
    return test_prediction


"for test purposes"
if __name__ == '__main__':
    training_lists = pickle_io.load_pickles('../data/training.dat')
    print("Successfully loaded training list")
    test_lists = pickle_io.load_pickles('../data/test.dat')
    print("Successfully loaded test list")
    print(knn_sk(test_lists, training_lists, 3, 10, 15))
