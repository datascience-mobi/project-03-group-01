import knn, pca, image_operations, load_image_vectors

tests_count = 0
tests_success = 0


def set_success_rate(prediction, test_image) -> None:
    global tests_count, tests_success
    if prediction == test_image.label:
        tests_success += 1
    tests_count += 1
    return None


def get_success_rate ():
    global tests_count, tests_success
    success = float(tests_success)/float(tests_count)
    return round(success, 4)


if __name__ == '__main__':
    k = 20  # number of nearest neighbors to check
    training_gz = load_image_vectors.load_gz('../data/mnist_train.csv.gz')
    training_lists = load_image_vectors.get_image_object_list(training_gz)
    print("Successfully loaded training list")
    test_gz = load_image_vectors.load_gz('../data/mnist_test.csv.gz')
    test_lists = load_image_vectors.get_image_object_list(test_gz)
    print("Successfully loaded test list")

    reduced_stuff = pca.prepare_data(training_lists[1], test_lists[1])
    for i in range(len(training_lists[0])):
        print(len(training_lists[0][i].image))
        training_lists[0][i].image = reduced_stuff[0][i]
        print(len(training_lists[0][i].image))
    for i in range(len(test_lists[0])):
        print(len(test_lists[0][i].image))
        test_lists[0][i].image = reduced_stuff[1][i]
        print(len(test_lists[0][i].image))

    # reduced_stuff[0] is train_list, [1] is test_list without digits

    # training_csv = load_image_vectors.load_csv('../data/mnist_train.csv') # Alternative to load_gz
    # for i in range (10):
    sorted_distances = knn.get_sorted_distances(test_lists[0][5], training_lists[0])
    # print("Successfully calculated distance of one test image to all training images")
    predicted_digit = knn.knn_distance_prediction(sorted_distances,k)
    print(predicted_digit, test_lists[0][5].label)
    #     set_success_rate(predicted_digit, test_list[i])
    # print("Success rate: " + str(get_success_rate()))