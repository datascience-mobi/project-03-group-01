from sklearn.decomposition import PCA
from sklearn import preprocessing
import src.image_operations as image_operations
import numpy


def increase_dimensions(pixellist):
    pixellist = numpy.array(pixellist)
    pca = PCA()
    pca.fit(pixellist)
    approximation = pca.inverse_transform(pixellist)
    # red = approximation.tolist()
    print(len(approximation))
    print(len(approximation[0]))
    image_operations.draw(approximation[0])


def reduce_dimensions(train_list, test_list) -> tuple:
    """
    Performs pca
    :param train_list: train list
    :param test_list: test list
    :return: reduced input lists as tuple
    """
    # # Scaler object for preprocessing input lists TODO mean normalizes?
    # scaler = preprocessing.StandardScaler()
    #
    # # Fit on training set only.
    # scaler.fit(train_list)
    #
    # # Apply transform to both the training set and the test set.
    # train_list = scaler.transform(train_list)
    # test_list = scaler.transform(test_list)
    #
    # # performs the dimension reduction itself
    # pca = PCA(.95)  # how much variance to keep
    #
    # # # Normalize around mean again
    # # pca.fit(train_list)
    # # print(pca.n_components_)
    # # train_list = pca.transform(train_list)
    # # test_list = pca.transform(test_list)
    #
    # test_list = pca.fit_transform(test_list)
    # # Lower dimension data is 5000x353 instead of 5000x1024
    # # lower_dimension_data.shape
    #
    # test_list.shape
    # # Project lower dimension data onto original features
    # approximation = pca.inverse_transform(test_list)
    # # Approximation is 5000x1024
    # approximation.shape
    # # Reshape approximation and X_norm to 5000x32x32 to display images
    # #approximation = approximation.reshape(-1, 32, 32)
    # #X_norm = X_norm.reshape(-1, 32, 32)
    #
    # #red = pca.inverse_transform(test_list)
    # for i in range(10):
    #     image_operations.draw(approximation[i])

    # For debugging: how many dimensions are kept
    # for i in range(5):
    #     print(len(train_list[i]))

    # For debugging: was every image kept?
    # print(len(train_list))
    # print(len(test_list))

    # return both lists separately as tuple

    # Should this variable be X_train instead of Xtrain?
    #X_train = numpy.random.randn(100, 50)

    scaler = preprocessing.StandardScaler()

    # Fit on training set only.
    scaler.fit(train_list)

    # Apply transform to both the training set and the test set.
    train_list = scaler.transform(train_list)
    test_list = scaler.transform(test_list)


    X_train = train_list
    X_test = test_list
    pca = PCA(0.95)
    pca.fit(X_train)

    # U, S, VT = np.linalg.svd(X_train - X_train.mean(0))
    #
    # assert_array_almost_equal(VT[:30], pca.components_)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # X_train_pca2 = (X_train - pca.mean_).dot(pca.components_.T)

    # assert_array_almost_equal(X_train_pca, X_train_pca2)

    # pca.ited in

    X_projected = pca.inverse_transform(X_train_pca)
    # X_projected2 = X_train_pca.dot(pca.components_) + pca.mean_

    # assert_array_almost_equal(X_projected, X_projected2)
    image_operations.draw(X_projected[0])

    return train_list, test_list
