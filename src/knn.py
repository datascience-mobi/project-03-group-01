def get_sorted_distances(test_image, training_list) -> list:
    """
    Calculates list containing the distance of each training vector to a certain test vector
    :param test_image: One CsvImage object containing the test image
    :param training_list: list of all CsvImage objects of the training images
    :return: sorted list of distances between the test image and each training image
    """

    distance_list = list()
    for training_image in training_list:
        distance = 0
        # print(training_image.image == test_image.image)
        for i in range(len(training_image.image)):
            difference = test_image.image[i]-training_image.image[i]
            square_distance = difference * difference
            distance += square_distance
        distance_list.append([distance, training_image.label])

    """ Test if list is sorted """
    # print("Distance vector length: " + str(len(distance_list)))

    """for i in range(5):
        print(distance_list[i])"""

    """ Sort list in ascending order """
    distance_list.sort()

    """ Test if list now is sorted"""
    """for i in range(5):
        print(distance_list[i])"""

    return distance_list


def knn_distance_prediction(results, k) -> int:
    """
    Calculates most frequent out of k lowest distances
    :param results: distance vector, containing [distance, label]
    :param k: value for k (k-nearest neighbors)
    :return: predicted / recognized digit
    """
    """ Create list only containing labels or training vectors, not distances """
    digits = list()
    if k <= len(results):
        for i in range(k):
            digits.append(results[i][1])
    print(digits)

    """ Get most frequent training image label """
    num_max_matches = 0  # TODO: figure out variable naming convention
    digit_max_matches = -1 # Set to -1 to easily detect errors
    for i in range(10):
        freq = digits.count(i) # count of digit 0-9 in digits vector
        if freq > num_max_matches:
            num_max_matches = freq # replace num_max_matches if a more frequent nearest neighbor was found
            digit_max_matches = i
    print("Prediction:" + str(digit_max_matches))
    return digit_max_matches
