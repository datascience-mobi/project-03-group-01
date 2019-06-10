import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
# from pylab import reParams
import urllib
import sklearn
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import struct
" this function tranforms the data to the numpy array"
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


raw_train = read_idx("../data_for_sklearn_KNN/train-images.idx3-ubyte")
# print(raw_train)
# print(raw_train.shape
train_data = np.reshape(raw_train,(60000,28*28))
train_label= read_idx("../data_for_sklearn_KNN/train-labels.idx1-ubyte")
# print(train_data.shape)
test_label = read_idx("../data_for_sklearn_KNN/t10k-labels.idx1-ubyte")
raw_test = read_idx("../data_for_sklearn_KNN/t10k-images.idx3-ubyte")
test_data = np.reshape (raw_test, (60000, 28*28))

