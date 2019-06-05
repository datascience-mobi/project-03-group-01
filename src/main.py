import src.knn as knn
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
    k = 20  # number of nearest neighbors to check
    training_list = load_image_vectors.load_gz('../data/mnist_train.csv.gz')
    print("Successfully loaded training list")
    test_list = load_image_vectors.load_gz('../data/mnist_test.csv.gz')
    print("Successfully loaded test list")

    sorted_distances = knn.get_sorted_distances(test_list[8], training_list)
    print("Successfully calculated distance of one test image to all training images")
    predicted_digit = knn.knn_distance_prediction(sorted_distances, k)
    print(predicted_digit, test_list[8].label)
