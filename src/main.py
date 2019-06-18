import src.KNN_sklearn as KNN_sklearn
import src.load_image_vectors as load_image_vectors
import src.plot as plot

total_number = 0
success_number = 0


def get_success_rate():
    global total_number, success_number
    success = float(success_number) / float(total_number)
    return round(success, 4)


def sklearn_k_value_test(k_min, k_max):
    # creates a list in the form of [[k1, accuracy], [k2, accuracy], ...]
    # then plots this created list as a diagram
    global total_number, success_number
    k_accuracy = list()
    for k in range(k_min, k_max):
        prediction_list = KNN_sklearn.knn_sk(test_lists, training_lists, k, 10)
        for idx, prediction in enumerate(prediction_list):
            total_number += 1
            if prediction == test_lists[0][idx].label:
                success_number += 1
        print("Success rate = " + str(get_success_rate))
        k_accuracy.append([k, get_success_rate()])
        reset_rates()
    plot.plot_k_values(k_accuracy)


def reset_rates():
    global total_number, success_number
    success_number = 0
    total_number = 0


if __name__ == '__main__':
    # number of nearest neighbors to check

    # load training and test images
    training_lists = load_image_vectors.load_gz('../data/mnist_train.csv.gz')
    print("Successfully loaded training list")
    test_lists = load_image_vectors.load_gz('../data/mnist_test.csv.gz')
    print("Successfully loaded test list")

    sklearn_k_value_test(1, 5)
