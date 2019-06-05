import src.knn as knn
import src.pca as pca
import src.load_image_vectors as load_image_vectors

tests_count = 0
tests_success = 0


def set_success_rate(prediction, test_image) -> None:
    global tests_count, tests_success
    if prediction == test_image.label:
        tests_success += 1
    tests_count += 1
    return None


def get_success_rate():
    global tests_count, tests_success
    success = float(tests_success)/float(tests_count)
    return round(success, 4)


if __name__ == '__main__':
    # number of nearest neighbors to check
    k = 20

    # load training and test images
    training_lists = load_image_vectors.load_gz('../data/mnist_train.csv.gz')
    print("Successfully loaded training list")
    test_lists = load_image_vectors.load_gz('../data/mnist_test.csv.gz')
    print("Successfully loaded test list")

    # Get reduces training and test images as tuple - reduced_images[0] is train_list, [1] is test_list without digits
    reduced_images = pca.reduce_dimensions(training_lists[1], test_lists[1])
    print("PCA finished successfully")

    # Replace unreduced CsvImage vectors by reduced ones, for training and test images
    for i in range(len(training_lists[0])):
        # print(len(training_lists[0][i].image)) # for debugging
        training_lists[0][i].image = reduced_images[0][i]
        # print(len(training_lists[0][i].image))  # how many dimensions after reduction, slows script down
    for i in range(len(test_lists[0])):
        # print(len(test_lists[0][i].image))
        test_lists[0][i].image = reduced_images[1][i]
        # print(len(test_lists[0][i].image))  # how many dimensions after reduction, slows script down
    print("Replaced images by reduced images")

    # perform KNN for dimension reduced images (one test image)
    sorted_distances = knn.get_sorted_distances(test_lists[0][5], training_lists[0])
    print("Successfully calculated distance of one test image to all training images")
    predicted_digit = knn.knn_distance_prediction(sorted_distances, k)
    print(predicted_digit, test_lists[0][5].label)
