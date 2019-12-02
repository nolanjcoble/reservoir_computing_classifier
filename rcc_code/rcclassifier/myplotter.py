import numpy as np
import matplotlib.pyplot as plt


def my_plotter(data, classes):
    rows0 = np.where((classes == [1, 0]).all(axis=1))
    class0 = data[rows0]
    rows1 = np.where((classes == [0, 1]).all(axis=1))
    class1 = data[rows1]

    plt.plot(class0[:, 0], class0[:, 1], 'bo')
    plt.plot(class1[:, 0], class1[:, 1], 'ro')