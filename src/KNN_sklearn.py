import sklearn
import src.load_image_vectors as load_image_vectors
import src.pickle_operations as pickle_io
from sklearn import neighbors, metrics


"the function takes two tuples: train and test; N_neighbours for KNN; M_labels for output "
"it reads the labels of test images and creates the 784 dim. space,"
"in which the points from test_tuple will be entered."
" then the KNN using N nighbours  with the points from test_tuple will be performed"
"output - M_labels of test_tuple"


def knn_sk(test_tuple, train_tuple, n_neighbours, m_labels):
    train_labels = list()
    test_pred = list()
    # for i in range(len(train_tuple)):
    #     train_labels.append(train_tuple[i].label)
    # print(train_labels)
    #print (train_lists[:10][:10])
    # print (train_labels[:10])
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbours).fit([csv_image.image for csv_image in training_lists], [csv_image.label for csv_image in training_lists])
    # print(knn.predict(test_list[:10]))
    for j in range(m_labels):
        test_pred.append([csv_image.image for csv_image in training_lists][j])
    test_prediction = knn.predict(test_pred)
    return test_prediction
    # return type (train_labels), type (test_lists), type (train_lists)


"for test purposes"
if __name__ == '__main__':
    training_lists = pickle_io.load_pickles('../data/training.dat')
    print("Successfully loaded training list")
    test_lists = pickle_io.load_pickles('../data/test.dat')
    print("Successfully loaded test list")
    print(knn_sk(test_lists, training_lists, 3, 10))
