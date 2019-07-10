import src.knn as knn
import src.image_operations as image_operations
import src.pca as pca
import src.load_image_vectors as load_image_vectors
import src.pickle_operations as pickle_io
import numpy as np
import src.image_operations as image_io
import matplotlib.pyplot as plt
import src.meta_digit_operations as meta_digit
import src.knn_clustering as knn_clustering

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
    # print("Successfully compressed and stored pickles")

    # COMMENT OUT LINES ABOVE AFTER RUNNING ONCE, THEN ONLY RUN CODE BELOW

    # Open CsvImage lists from pickle files - lowers loading time by factor 10
    # loading from bz2: 15.75s, from uncompressed .dat: 4.458s
    training_lists = pickle_io.load_pickles("../data/training.dat")
    test_lists = pickle_io.load_pickles("../data/test.dat")
    print("Successfully loaded images from compressed pickle files")

    mean_digits = meta_digit.get_mean_digits(training_lists)
    median_digits = meta_digit.get_median_digits(training_lists)

    meta_digit.show_mean_digits(training_lists)
    inner_distance = list()
    for i in range(10):
        inner_distance.append(meta_digit.mean_center_distance([train.image for train in training_lists if train.label == i], mean_digits[i]))
    # min_dist_mean = 0
    heat_list_mean = list()
    for i in range(len(mean_digits)):
        for j in range(len(mean_digits)):
            dist = abs(meta_digit.mean_center_distance([train.image for train in training_lists if train.label == i], mean_digits[j]) - inner_distance[i])
            # if dist < min_dist_mean and not dist == 0:
            #     min_dist_mean = dist
            heat_list_mean.append(dist)
    min_dist_mean = min([heat for heat in heat_list_mean if not heat == 0])
    max_dist_mean = max(heat_list_mean)
    # a = np.random.random((16, 16))
    heat_list_mean = np.asarray(heat_list_mean)
    plt.imshow(heat_list_mean.reshape(10, 10), cmap='hot', interpolation='nearest')
    plt.xlabel(f"Distances between digit clusters to others's center \n Min: {min_dist_mean} \n Max: {max_dist_mean} \n Diff: {min_dist_mean/max_dist_mean}")
    plt.show()

    meta_digit.show_median_digits(training_lists)
    meta_digit.show_best_digits(training_lists, test_lists)
    meta_digit.show_as_heatmap(mean_digits, median_digits)
    knn_clustering.get_mispredictions(training_lists, test_lists)