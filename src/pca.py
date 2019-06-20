from sklearn.decomposition import PCA
from sklearn import preprocessing


def reduce_dimensions(train_list, test_list) -> tuple:
    """
    Performs pca
    :param train_list: train list
    :param test_list: test list
    :return: reduced input lists as tuple
    """
    # Scaler object for preprocessing input lists TODO mean normalizes?
    scaler = preprocessing.StandardScaler()

    # Fit on training set only.
    scaler.fit(train_list)

    # Apply transform to both the training set and the test set.
    train_list = scaler.transform(train_list)
    test_list = scaler.transform(test_list)

    # performs the dimension reduction itself
    pca = PCA(.95)  # how much variance to keep

    # Normalize around mean again
    pca.fit(train_list)
    print(pca.n_components_)
    train_list = pca.transform(train_list)
    test_list = pca.transform(test_list)

    # For debugging: how many dimensions are kept
    # for i in range(5):
    #     print(len(train_list[i]))

    # For debugging: was every image kept?
    # print(len(train_list))
    # print(len(test_list))

    # return both lists separately as tuple
    return train_list, test_list
