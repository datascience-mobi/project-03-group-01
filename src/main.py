import knn

if __name__ == '__main__':
    for i in range(100 ):
        knn.perform_knn(7,"../data/mnist_test.csv",i+1,"../data/mnist_train.csv")
    image_operations.save_as_image("../data/mnist_train.csv", 14)