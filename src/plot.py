import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np


def plot_values (input_list):
    # for training purposes

    labels, ys = zip(*a)
    xs = np.arange(len(labels))
    width = 0.8

    plt.bar(xs, ys, width, align='center')

    plt.xticks(xs, labels)
    # Replace default x-ticks with xs, then replace xs with labels
    plt.yticks(ys)

    plt.ylabel('Accuracy')
    plt.xlabel('#k')
    plt.title('Accuracy test')

    plt.show()

if __name__ == '__main__':
    a = [[1, 7], [2, 8], [3, 0], [4, 9], [5, 12]]
    plot_values(a)
