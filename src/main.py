import src.knn as knn
# import src.pca as pca
import src.image_operations as image_operations
import src.load_image_vectors as load_image_vectors
import src.drawing_canvas as drawing_canvas
import src.knn as knn
import src.pca as pca
import src.load_image_vectors as load_image_vectors
import src.pickle_operations as pickle_io

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
    # number of nearest neighbors to check
    k = 20

    # # TODO lists are currently a tuple of CsvImage Objects and the pure integer lists
    # # load training and test images - only necessary once combined with saving as pickle
    # training_lists = load_image_vectors.load_gz('../data/mnist_train.csv.gz')
    # print("Successfully loaded training list")
    # test_lists = load_image_vectors.load_gz('../data/mnist_test.csv.gz')
    # print("Successfully loaded test list")
    #
    # # Save created CsvImage lists in pickle files
    # pickle_io.save_compressed_pickles(training_lists, "../data/training.dat.bz2")
    # pickle_io.save_compressed_pickles(test_lists, "../data/test.dat.bz2")
    # print("Successfully compressed and stored pickles")

    # COMMENT OUT LINES ABOVE AFTER RUNNING ONCE, THEN ONLY RUN CODE BELOW

    # Open CsvImage lists from pickle files - lowers loading time by factor 10
    # loading from bz2: 15.75s, from uncompressed .dat: 4.458s
    training_lists = pickle_io.load_pickles("../data/training.dat")
    test_lists = pickle_io.load_pickles("../data/test.dat")
    print("Successfully loaded images from compressed pickle files")
    #
    # Get reduced training and test images as tuple - reduced_images[0] is train_list, [1] is test_list without digits
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
    predicted_digit = knn.knn_digit_prediction(test_lists[0][7], training_lists[0], k)
    print("Predicted digit: " + str(predicted_digit) + " , expected result: " + str(test_lists[0][7].label))

    # optionally save reference image for later inspection
    # image_operations.save('../fname.png', test_list[1].image)

    # get drawn image, adjusted for mnist
    test_vector = drawing_canvas.drawn_image()

    # insert label to fit test_vector for CsvImage class
    test_vector.insert(0, -1)

    # Creates CsvImage object for drawn test image
    test = load_image_vectors.CsvImage(test_vector, is_list=True)

    # perform KNN
    sorted_distances = knn.get_sorted_distances(test, training_list)
    # print("Successfully calculated distance of one test image to all training images")
    predicted_digit = knn.knn_distance_prediction(sorted_distances, k)
    print(predicted_digit)
