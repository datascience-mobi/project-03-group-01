# from pylab import reParams
import sklearn

import src.load_image_vectors as load_image_vectors
from sklearn import neighbors, metrics

"the function takes two tuples: train and test; N_neighbours for KNN; M_labels for output "
"it reads the labels of test images and creates the 784 dim. space,"
"in which the points from test_tuple will be entered."
" then the KNN using N nighbours  with the points from test_tuple will be performed"
"output - M_labels of test_tuple"


def knn_sk(test_tuple, train_tuple, n_neighbours, m_labels):
    train_labels = list()
    test_pred = list()
    for i in range(len(train_tuple[0])):
        train_labels.append(train_tuple[0][i].label)
    #print (train_lists[:10][:10])
    # print (train_labels[:10])
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbours).fit(train_tuple[1], train_labels)
    # print(knn.predict(test_list[:10]))
    for j in range(m_labels):
        test_pred.append(test_tuple[1][j])
    test_prediction = knn.predict(test_pred)
    return test_prediction
    # return type (train_labels), type (test_lists), type (train_lists)


"for test purposes"

training_lists = load_image_vectors.load_gz('../data/mnist_train.csv.gz')
print("Successfully loaded training list")
test_lists = load_image_vectors.load_gz('../data/mnist_test.csv.gz')
print("Successfully loaded test list")

# def dim(a):
#     if not type(a) == list:
#         return []
#     return [len(a)] + dim(a[0])


# print(dim(test_lists))


print(knn_sk(test_lists, training_lists, 3, 10))
#print (type(training_lists))
#def read_idx(filename):


#     with open(filename, 'rb') as f:
#         zero, data_type, dims = struct.unpack('>HBB', f.read(4))
#         shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
#         return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


# raw_train = read_idx("../data_for_sklearn_KNN/train-images.idx3-ubyte")
# # print(raw_train)
# # print(raw_train.shape
# train_data = np.reshape(raw_train,(60000,28*28))
# train_label= read_idx("../data_for_sklearn_KNN/train-labels.idx1-ubyte")
# # print(train_data.shape)
# test_label = read_idx("../data_for_sklearn_KNN/t10k-labels.idx1-ubyte")
# raw_test = read_idx("../data_for_sklearn_KNN/t10k-images.idx3-ubyte")
# test_data = np.reshape (raw_test, (10000, 28*28))
