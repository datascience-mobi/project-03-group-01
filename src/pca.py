from sklearn.decomposition import PCA
from sklearn import preprocessing
import src.image_operations as image_operations
import numpy


def increase_dimensions(train_list, reduced_images, original_dimensions):
    """
    Reconstructs visible 28×28 images from dimension reduced ones
    :param train_list: preprocessed training images -> no scaler object necessary
    :param reduced_images: dimension reduced images created by reduce_dimensions
    :param original_dimensions: number of dimensions the images were reduced to
    :return: numpy array of one reconstructed image
    """
    pca = PCA(n_components=original_dimensions)
    pca.fit(train_list)
    approximation = pca.inverse_transform(reduced_images)
    new_image = approximation[63]
    new_image = numpy.interp(new_image, (new_image.min(), new_image.max()), (0, 255))

    # # For debug: min and max values after scaling
    # print(min(new_image))
    # print("--")
    # print(max(new_image))

    # # Draw scaled image
    # new_image.tolist()
    # new_image = [round(x) for x in new_image]
    # image_operations.draw(new_image)

    return numpy.around(new_image)


def reduce_dimensions(train_list, test_list, target_dimensions) -> tuple:
    """
    Performs pca
    :param train_list: train list
    :param test_list: test list
    :param target_dimensions: number of dimensions to reduce to
    :return: reduced input lists as tuple
    """
    test_index = 63

    image_operations.draw(test_list[test_index])

    scaler = preprocessing.StandardScaler()

    # Fit on training set only.
    scaler.fit(train_list)

    # Apply transform to both the training set and the test set.
    train_list = scaler.transform(train_list)
    test_list = scaler.transform(test_list)

    # Create instance of pca and fit it only to the training images
    pca = PCA(n_components=target_dimensions)
    pca.fit(train_list)

    # Apply pca to both image lists
    train_pca = pca.transform(train_list)
    test_pca = pca.transform(test_list)

    return train_pca, test_pca, train_list
