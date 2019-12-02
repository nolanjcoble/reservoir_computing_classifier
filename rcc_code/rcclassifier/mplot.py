from rcclassifier.ReservoirComputer import ReservoirComputer as rc
import numpy as np
import matplotlib.pyplot as plt


def mplot(num_classes, data, classes):
    separated_data = rc.get_separated_by_class(data, classes, num_classes)

    for i in range(num_classes):
        cur_data = separated_data[:, :, i]
        cur_data = cur_data[~np.all(cur_data == 0, axis=1)]
        plt.plot(cur_data[:, 0], cur_data[:, 1], "o")
