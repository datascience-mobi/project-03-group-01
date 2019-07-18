from scipy.spatial import distance
from src import load_image_vectors

import numpy as np
from src import pickle_operations as pickle_io
from src import digit_evaluation as mdo
import random


def get_sorted_distances_from_testpool_to_meta(meta_digit, training_list, meta_digit_label) -> list:
    """
    :param meta_digit:
    :param training_list: list of 60000 csv images (test pool)
    :param meta_digit_label:
    :return:
    """
    # Scipy function calculating the euclidean distance between the test image and all training images
    distance_list = list()
    for training in training_list:
        if training.label == meta_digit_label:
            dist = distance.euclidean(training.image, meta_digit)
            distance_list.insert(0, dist)

    distance_list.sort()

    return distance_list


def getting_random_digit_from_test_pool(test_images, label_of_meta_digit):
    test_images_needed_label = list()
    for test in test_images:
        if test.label == label_of_meta_digit:
            test_images_needed_label.append(test)
    random_test_image = test_images_needed_label[random.randrange(0, len(test_images_needed_label) - 1)]
    return random_test_image


def get_distance_from_handwritten_to_meta(hand_written_digit, meta_digit):
    dist = list()
    dist = distance.euclidean(hand_written_digit, meta_digit)
    return dist


def percentile_of_handwritten_digit(distance_between_handwritten_and_meta, sorted_distances_from_meta):
    """
    Function prints the evaluation dependent on the percentile of hand written digit.
    :param distance_between_handwritten_and_meta:
    :param sorted_distances_from_meta:
    :return: percentile
    """
    for i in range(len(sorted_distances_from_meta)):
        if sorted_distances_from_meta[i] > distance_between_handwritten_and_meta:
            percent = round(((len(sorted_distances_from_meta) - i) / len(sorted_distances_from_meta) * 100), 3)
            break
    print("your written digit is closer to the most recognisable digit than", percent, "% of digits from our database")
    if percent < 30:
        evaluation_feedback = "you could do better"
        # print("you could do better")
    elif percent < 80:
        evaluation_feedback = "is recognisable, you did fine"
        # print("is recognisable, you did fine")
    else:
        evaluation_feedback = "really well-done (written)"
        # print("really well-done (written)")
    return [percent, evaluation_feedback]


# load training and test images
# training_lists = load_image_vectors.load_gz('../data/mnist_train.csv.gz')
# print("Successfully loaded training list")
# test_lists = load_image_vectors.load_gz('../data/mnist_test.csv.gz')
# print("Successfully loaded test list")

# pickle_io.save_pickles(training_lists, "../data/training.dat")
# pickle_io.save_pickles(test_lists, "../data/test.dat")
# print("Successfully compressed and stored pickles")

# training_lists = pickle_io.load_pickles("../data/training.dat")
# test_lists = pickle_io.load_pickles("../data/test.dat")
# print("Successfully loaded images from compressed pickle files")
#
# # average square distance between metadigits and all elements with the same label
# median_digit = mdo.get_median_digits(training_lists)
# mean_digit = mdo.get_mean_digits(training_lists)
# best_digit = mdo.get_best_digits(training_lists, test_lists)
# distance_sum = list()
# distance_average = list()
# sum_of_distances = float
# average_for_every_digit_median = list()
# average_for_every_digit_mean = list()
# average_for_every_digit_best = list()
#
# # calculating the average squared distances for median digit
# for i in range(10):
#     distances_for_sum = get_sorted_distances_from_testpool_to_meta(median_digit[i], training_lists, i)
#     sum_of_distances = sum(distances_for_sum)
#     # for j in range(len(distances_for_sum)-1):
#     #     sum_of_distances = sum_of_distances + distances_for_sum[j]
#     average_for_every_digit_median.append(sum_of_distances / len(distances_for_sum))
# print("average distances to the median digit")
# print(average_for_every_digit_median)
#
# # calculating the average squared distances for mean digit
# for i in range(10):
#     distances_for_sum = get_sorted_distances_from_testpool_to_meta(mean_digit[i], training_lists, i)
#     sum_of_distances = sum(distances_for_sum)
#     # for j in range(len(distances_for_sum)-1):
#     #     sum_of_distances = sum_of_distances + distances_for_sum[j]
#     average_for_every_digit_mean.append(sum_of_distances / len(distances_for_sum))
# print("average distances to the mean digit")
# print(average_for_every_digit_mean)
#
# # calculating the average squared distances for best digit
# for i in range(10):
#     distances_for_sum = get_sorted_distances_from_testpool_to_meta(best_digit[i].image, training_lists, i)
#     sum_of_distances = sum(distances_for_sum)
#     # for j in range(len(distances_for_sum)-1):
#     #     sum_of_distances = sum_of_distances + distances_for_sum[j]
#     average_for_every_digit_best.append(sum_of_distances / len(distances_for_sum))
# print("average distances to the best digit")
# print(average_for_every_digit_best)

def get_evaluation(best_digit, training_lists, drawn_digit, label):
    distance_btw_handwritten_and_meta = get_distance_from_handwritten_to_meta(drawn_digit.image, best_digit.image)
    sorted_distances_from_meta = get_sorted_distances_from_testpool_to_meta(best_digit.image, training_lists, label)
    evaluation = percentile_of_handwritten_digit(distance_btw_handwritten_and_meta, sorted_distances_from_meta)
    return evaluation

# written_digit = getting_random_digit_from_test_pool(test_lists, 0)
# distance_btw_handwritten_and_meta = get_distance_from_handwritten_to_meta(written_digit.image, best_digit[0].image)
# sorted_distances_from_meta = get_sorted_distances_from_testpool_to_meta(best_digit[0].image, training_lists, 0)
# evaluation = percentile_of_handwritten_digit(distance_btw_handwritten_and_meta, sorted_distances_from_meta)
# print(evaluation)
