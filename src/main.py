import knn

if __name__ == '__main__':
    for i in range(50):
        knn.perform_knn(7,"../data/mnist_test.csv",i+10,"../data/mnist_train.csv")