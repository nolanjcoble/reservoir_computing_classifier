# import statements
from rcclassifier.ReservoirComputer import ReservoirComputer as rc
import numpy as np
from sklearn import model_selection
import time
import matplotlib.pyplot as plt
from rcclassifier.myplotter import my_plotter as mplot
from run_classifier import get_params, format_classes

if __name__ == '__main__':
    sz = 200
    # make two distinct classes
    testers = np.random.randint(2, size=sz)
    data = np.transpose(np.vstack((testers, testers)))
    print(data)

    # randomly adjust the values
    adjust = (np.random.rand(sz, 2) - 0.5)*0.5
    data = data + adjust

    un_classes = np.zeros((sz, 1))
    # go through and assign random classes
    for idx, _ in enumerate(data):
        un_classes[idx] = round(np.random.rand())

    classes = format_classes(un_classes.astype(int))

    # get the parameters
    training_length, num_properties = data.shape
    num_classes = np.unique(classes).size

    properties = {
        "reservoir_size": 100,
        "training_length": training_length,
        "num_properties": num_properties,
        "num_classes": num_classes
    }
    resparams = get_params(**properties)  # parameters for the reservoir layers

    # generate the reservoir computer
    classifier = rc(resparams)

    plt.figure()
    mplot(data, classes)
    plt.show()
    for x in range(20):
        # train the classifier
        classifier.train_reservoir(data, classes)

        classes = classifier.rc_classification(data)
        mplot(data, classes)
        plt.show()

    mplot(data, classes)
    plt.show()

