from sklearn.decomposition import PCA
from sklearn import preprocessing
import src.image_operations as image_operations
import numpy
import matplotlib.pyplot as plt


# Does not work with a specified variance value, only produces noise without specified variance
# to execute, de-comment and run (only) lines 83 - 89 in main.py
def increase_dimensions(train_list, pixellist):
    scaler = preprocessing.StandardScaler()

    # Fit on training set only.
    scaler.fit(train_list)

    # Apply transform to both the training set and the test set.
    train_list = scaler.transform(train_list)

    pca = PCA(0.95)
    pca.fit(train_list)
    approximation = pca.inverse_transform(pixellist)
    # red = approximation.tolist()
    new_image  =approximation[61]
    new_image = numpy.interp(new_image, (new_image.min(), new_image.max()), (0, 255))

    # For debug: min and max values after scaling
    print(min(new_image))
    print("--")
    print(max(new_image))

    # Draw scaled image
    new_image.tolist()
    new_image = [round(x) for x in new_image]
    image_operations.draw(new_image)


# performs pca and inverts it. draws 3 images: before pca, after inversion, after inversion with scaling values to 0-255
# digits (subjectively) recognizable with ~ 30%+ variance, but digit partly indistinguishable from background
# to execute, run lines 55 - 63 in main.py
def reduce_dimensions(train_list, test_list) -> tuple:
    """
    Performs pca
    :param train_list: train list
    :param test_list: test list
    :return: reduced input lists as tuple
    """
    test_index = 61

    image_operations.draw(test_list[test_index])

    scaler = preprocessing.StandardScaler()

    # Fit on training set only.
    scaler.fit(train_list)

    # Apply transform to both the training set and the test set.
    train_list = scaler.transform(train_list)
    test_list = scaler.transform(test_list)

    # Create instance of pca and fit it only to the training images
    pca = PCA(.95)
    pca.fit(train_list)

    # Apply pca to both image lists
    train_pca = pca.transform(train_list)
    test_pca = pca.transform(test_list)

    return train_pca, test_pca
