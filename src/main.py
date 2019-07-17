from random import randint
from scipy.spatial import distance
import src.drawing_canvas as drawing_canvas
import src.knn as knn
import src.image_operations as image_operations
import src.pca as pca
import src.load_image_vectors as load_image_vectors
import src.pickle_operations as pickle_io
import src.image_operations as image_io
import src.meta_digit_operations as meta_digit
import src.digit_evaluation as digit_evaluation
import numpy as np


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
    print("Successfully loaded images from compressed pickle files")

    # optionally save reference image for later inspection
    # image_operations.save('../fname.png', test_list[1].image)

    while True:
        # get drawn image, adjusted for mnist
        random_digit = randint(0, 9)
        print(random_digit)

        test_vector = drawing_canvas.drawn_image(random_digit)
        if all([v == 0 for v in test_vector]):
            break

        # test_vector = drawing_canvas.image_prepare_old("mnist.png")

        # insert label to fit test_vector for CsvImage class
        test_vector.insert(0, -1)

        # Creates CsvImage object for drawn test image
        test = load_image_vectors.CsvImage(test_vector, is_list=True)

        # perform KNN
        predicted_digit = knn.knn_digit_prediction(test, training_lists, k)
        print(predicted_digit)

        best_digit = digit_evaluation.get_best_digits(training_lists, test_lists)
        evaluation = -1
        digit_evaluation.show_difference(np.asarray(test.image), random_digit, np.asarray(best_digit[random_digit].image), evaluation)