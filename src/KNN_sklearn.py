import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, metrics
from src import load_image_vectors


def knn_sk(train_images, test_images, n_neighbours):
    test_pred = list()
    # define knn function using train data as a space for prediction
    # the function needs two lists: of data points in 784 dim. space and of their labels
    # they are written in addition .fit ()
    # n_neighbours defines how many neighbours we want to take while making prediction
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbours).fit([csv_image.image for csv_image in train_images], [csv_image.label for csv_image in train_images])
    # loop for prediction of only specific wanted digits between min_index and max_index-1
    # the actual points of test data set are extracted as test_pred
    test_pred = knn.predict([csv_image.image for csv_image in test_images])
    for i in range(len(test_images)):
        test_images[i].set_prediction(test_pred[i])
    return test_images


if __name__ == '__main__':
    training_lists = load_image_vectors.load_gz('../data/mnist_train.csv.gz')
    print("Successfully loaded training list")
    test_lists = load_image_vectors.load_gz('../data/mnist_test.csv.gz')
    print("Successfully loaded test list")
    predictions = knn_sk(training_lists, test_lists, 3)
    for j in range(10):
        print(predictions[j].prediction)
