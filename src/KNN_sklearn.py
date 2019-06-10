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
# print(raw_train.shape)
raw_data = np.reshape(raw_train,(60000,28*28))
