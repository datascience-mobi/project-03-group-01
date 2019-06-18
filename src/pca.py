from sklearn.decomposition import PCA
from sklearn import preprocessing
import src.image_operations as image_operations
import numpy


# Does not work with a specified variance value, only produces noise without specified variance
# to execute, de-comment and run (only) lines 83 - 89 in main.py
def increase_dimensions(pixellist):
    pixellist = numpy.array(pixellist)
    pca = PCA(n_components=12)
    pca.fit(pixellist)
    approximation = pca.inverse_transform(pixellist)
    # red = approximation.tolist()
    print(len(approximation))
    print(len(approximation[0]))
    image_operations.draw(approximation[0])


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
    import matplotlib.pyplot as plt
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

    # Reconstruct one test image from its reduced form
    test_inverse = pca.inverse_transform(test_pca)

    plt.figure(figsize=(8, 4))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(test_list[test_index].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 255))
    plt.xlabel('784 components', fontsize=14)
    plt.title('Original Image', fontsize=20)

    # 154 principal components
    plt.subplot(1, 2, 2)
    plt.imshow(test_inverse[test_index].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 255))
    plt.xlabel('154 components', fontsize=14)
    plt.title('95% of Explained Variance', fontsize=20)

    # Draw image as it was created by inverse_transform
    image_operations.draw(test_inverse[test_index])

    # For debug: min and max values of reconstructed image
    print(min(test_inverse[test_index]))
    print("--")
    print(max(test_inverse[test_index]))

    new_image = test_inverse[test_index]
    # Scale reconstructed image values to a range from 0 to 255
    new_image = numpy.interp(new_image, (new_image.min(), new_image.max()), (0, 255))

    # For debug: min and max values after scaling
    print(min(new_image))
    print("--")
    print(max(new_image))

    # Draw scaled image
    new_image = [round(x) for x in new_image]
    image_operations.draw(new_image)

    # # TODO perform pca for 100% variance, remove k least variant components and revert pca
    # pca = PCA(n_components=50)
    # test_pca = numpy.delete(test_pca, 50, 1)
    #
    # # Reconstruct one test image from its reduced form
    # test_inverse = pca.inverse_transform(test_pca)
    #
    # # Draw image as it was created by inverse_transform
    # image_operations.draw(test_inverse[test_index])
    #
    # # For debug: min and max values of reconstructed image
    # print(min(test_inverse[test_index]))
    # print("--")
    # print(max(test_inverse[test_index]))
    #
    # new_image = test_inverse[test_index]
    # # Scale reconstructed image values to a range from 0 to 255
    # new_image = numpy.interp(new_image, (new_image.min(), new_image.max()), (0, 255))
    #
    # # For debug: min and max values after scaling
    # print(min(new_image))
    # print("--")
    # print(max(new_image))

    return train_pca, test_pca
