from sklearn.decomposition import PCA
from sklearn import preprocessing
import src.image_operations as image_operations
import numpy
import matplotlib.pyplot as plt


def plot_inverse_transforms(train_list, reduced_images, scaler):
    # Invert pca for multiple values and draw the yielded images to one plot
    for idx, i in enumerate([10, 20, 40, 70, 100, 200, 400, 784]):
        image = increase_dimensions(train_list, [red[:i] for red in reduced_images], i, scaler)
        plt.subplot(2, 4, idx+1)
        plt.imshow(image.reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest',
                   clim=(0, 255))
        plt.xlabel(f"n(dim) = {i}", fontsize=14)
    plt.show()


def increase_dimensions(train_list, reduced_images, original_dimensions, scaler):
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
    approximation = scaler.inverse_transform(approximation)
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

    return train_pca, test_pca, train_list, scaler
