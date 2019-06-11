import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
# from pylab import reParams
import urllib
import sklearn
import src.load_image_vectors as load_image_vectors
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import struct


# noinspection PyUnresolvedReferences
def knn_sk(test_lists, train_lists):

    for i in range(60000):
        train_labels[i][0] = train_lists [i][0]
        train_image[i][0:] = train_lists [i][1:]

    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3).fit(train_image, train_labels)
    # print(knn.predict(test_list[:10]))
    test_prediction = knn.predict(test_lists)
    return test_prediction


"for test purposes"
# training_lists = load_image_vectors.load_gz('../data/mnist_train.csv.gz')
# print("Successfully loaded training list")
# test_lists = load_image_vectors.load_gz('../data/mnist_test.csv.gz')
# print("Successfully loaded test list")
#
#
# print(knn_sk(test_lists,training_lists)[:10])

                                                                                                                                        # def read_idx(filename):
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


