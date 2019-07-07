import src.knn as knn
import src.pca as pca
import src.pickle_operations as pickle_io


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
    # pickle_io.save_pickles(training_lists, "../data/training.dat")
    # pickle_io.save_pickles(test_lists, "../data/test.dat")
    # print("Successfully stored pickles")

    # COMMENT OUT LINES ABOVE AFTER RUNNING ONCE, THEN ONLY RUN CODE BELOW

    # Open CsvImage lists from pickle files - lowers loading time by factor 10
    # loading from bz2: 15.75s, from uncompressed .dat: 4.458s
    training_lists = pickle_io.load_pickles("../data/training.dat")
    test_lists = pickle_io.load_pickles("../data/test.dat")
    print("Successfully loaded images from pickle files")

    # Get reduces training and test images as tuple - reduced_images[0] is train_list, [1] is test_list without digits
    reduced_images = pca.reduce_dimensions([csv_image.image for csv_image in training_lists], [csv_image.image for csv_image in test_lists], 784)
    print("PCA finished successfully")
    for i in [10, 20, 50, 100, 200, 400, 784]:
        pca.increase_dimensions(reduced_images[2], [red[:i] for red in reduced_images[1]], i)
    #
    # Replace unreduced CsvImage vectors by reduced ones, for training and test images
    for i in range(len(training_lists)):
        # print(len(training_lists[0][i].image)) # for debugging
        training_lists[i].image = reduced_images[0][i]
        # print(len(training_lists[0][i].image))  # how many dimensions after reduction, slows script down
    for i in range(len(test_lists)):
        # print(len(test_lists[0][i].image))
        test_lists[i].image = reduced_images[1][i]
        # print(len(test_lists[0][i].image))  # how many dimensions after reduction, slows script down
    print("Replaced images by reduced images")

    # # Save created CsvImage lists in pickle files
    # pickle_io.save_pickles(training_lists, "../data/red_training.dat")
    # pickle_io.save_pickles(test_lists, "../data/red_test.dat")
    #
    # # # Open CsvImage lists from pickle files - lowers loading time by factor 10
    # training_lists = pickle_io.load_pickles("../data/red_training.dat")
    # test_lists = pickle_io.load_pickles("../data/red_test.dat")
    # print("Successfully loaded images from pickle files")

    # perform KNN for dimension reduced images (one test image)
    predicted_digit = knn.knn_digit_prediction(test_lists[7], training_lists, k)
    print("Predicted digit: " + str(predicted_digit) + " , expected result: " + str(test_lists[7].label))
