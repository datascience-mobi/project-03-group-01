import knn, pca, image_operations

if __name__ == '__main__':
    #for i in range(5):
     #   knn.perform_knn(7,"../data/mnist_test.csv",i+1,"../data/mnist_train.csv")
    #image_operations.save_as_image("../data/mnist_train.csv", 14)
    pca.perform_pca("../data/mnist_train.csv")
