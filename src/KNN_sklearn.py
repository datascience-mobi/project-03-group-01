# from pylab import reParams
import sklearn
import src.load_image_vectors as load_image_vectors
from sklearn import neighbors, metrics


def knn_sk(test_tuple, train_tuple, n_neighbours, m_labels):
    # the function takes two tuples: train and test; N_neighbours for KNN; M_labels for output
    # it reads the labels of test images and creates the 784 dim. space,
    # in which the points from test_tuple will be entered.
    #  then the KNN using N neighbours  with the points from test_tuple will be performed
    # output - M_labels of test_tuple

    train_labels = list()
    test_pred = list()
    for i in range(len(train_tuple[0])):
        train_labels.append(train_tuple[0][i].label)
    # print (train_lists[:10][:10])
    # print (train_labels[:10])
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbours).fit(train_tuple[1], train_labels)
    # print(knn.predict(test_list[:10]))
    for j in range(m_labels):
        test_pred.append(test_tuple[1][j])
    test_prediction = knn.predict(test_pred)
    return test_prediction
    # return type (train_labels), type (test_lists), type (train_lists)


