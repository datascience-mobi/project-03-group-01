import sklearn


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
    # define knn function using train data as a space for prediction
    # the function needs two lists: of data points in 784 dim. space and of their labels
    # they are written in addition .fit ()
    # n_neighbours defines how many neighbours we want to take while making prediction
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbours).fit([csv_image.image for csv_image in train_images], [csv_image.label for csv_image in train_images])
    # loop for prediction of only specific wanted digits between min_index and max_index-1
    # the actual points of test data set are extracted as test_pred
    for j in range(min_index, max_index):
        test_pred.append([csv_image.image for csv_image in test_images][j])
        print(j)
    # application of knn to the extracted test points and converting to list
    test_prediction = knn.predict(test_pred).tolist()
    # creating 2D list by pairing the elements of two tuples: range of wanted indexes and test_prediction
    # 2d list has this form [index/position of the digit1][prediction made by knn],... for every digits in a given range
    # this list is our output
    test_prediction = [list(a) for a in zip([j for j in range(min_index, max_index)], test_prediction)]
    return test_prediction
