from anneal import SimAnneal
import numpy as np
import matplotlib.pyplot as plt
import random

sz = 100
# make two distinct classes
testers = np.random.randint(2, size=sz)
data = np.transpose(np.vstack((testers, testers)))
print(data)

# randomly adjust the values
adjust = (np.random.rand(sz, 2) - 0.5)*0.5
data = data + adjust

"""
# specify document name
file_name = "multivar_data_nb.txt"

all_data = np.loadtxt(file_name, delimiter=',')

# last column of data contains the classes for each input vector
data = all_data[:, :-1]
"""
# get the parameters
training_length, num_properties = data.shape

properties = {
    "reservoir_size": 20,
    "training_length": training_length,
    "num_properties": num_properties,
    "num_classes": 2
}


if __name__ == "__main__":
    sa = SimAnneal(data, properties, stopping_iter=5000)
    sa.anneal()
    plt.figure()
    sa.plot_best()
    plt.show()
    plt.figure()

