import src.knn as knn
# import src.pca as pca
import src.image_operations as image_operations
import src.load_image_vectors as load_image_vectors
import src.drawing_canvas as drawing_canvas

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
    training_list = load_image_vectors.get_image_object_list(training_gz)
    print("Successfully loaded training list")
    # test_gz = load_image_vectors.load_gz('../data/mnist_test.csv.gz')
    # test_list = load_image_vectors.get_image_object_list(test_gz)
    # print("Successfully loaded test list")
    #
    # print(test_list[1].image)
    image_operations.draw(training_list[6].image)
    # image_operations.save('../fname.png', test_list[1].image)
    test_vector = drawing_canvas.drawn_image()
    test_vector.insert(0,-1)
    test = load_image_vectors.CsvImage(test_vector, is_list=True)
    sorted_distances = knn.get_sorted_distances(test, training_list)
    # print("Successfully calculated distance of one test image to all training images")
    predicted_digit = knn.knn_distance_prediction(sorted_distances,k)
    print(predicted_digit)

    # training_csv = load_image_vectors.load_csv('../data/mnist_train.csv') # Alternative to load_gz
    #for i in range (10):
    #    sorted_distances = knn.get_sorted_distances(test_list[i], training_list)
    #    # print("Successfully calculated distance of one test image to all training images")
    #    predicted_digit = knn.knn_distance_prediction(sorted_distances,k)
    #    set_success_rate(predicted_digit, test_list[i])
    #print("Success rate: " + str(get_success_rate()))