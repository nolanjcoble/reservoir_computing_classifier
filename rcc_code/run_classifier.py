# import statements
from rcclassifier.ReservoirComputer import ReservoirComputer as rc
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
import time
import matplotlib.pyplot as plt
from rcclassifier.mplot import mplot


def get_data(file_name, split=0.25):
    """

    Parameters
    ----------
    file_name: name of file to be read
    split: percentage of data to use as training (default is 0.25)

    Returns
    -------
    training_data:
        split part of the training
    training_classes:
        corresponding classes
    test_data:
        rest of data for testing
    test_classes:
        corresponding classes
    """

    # load the data from the file
    data = np.loadtxt(file_name, delimiter=',')

    # last column of data contains the classes for each input vector
    all_data, unformatted_classes = data[:, :-1], data[:, -1]

    # adjust the classes to be formatted properly
    all_classes = format_classes(unformatted_classes.astype(int))

    # split the data and classes into test and training
    training_data, test_data, training_classes, test_classes = \
        model_selection.train_test_split(all_data, all_classes, test_size=split, random_state=5)

    return training_data, test_data, training_classes, test_classes


def get_params(num_properties, num_classes, training_length, reservoir_size=10):
    """

    Parameters
    ----------
    num_properties: int
        number of properties per input vector
    num_classes: int
        number of classes that are present
    training_length: int
        number of input vectors
    reservoir_size: int
        size of reservoir

    Returns
    -------
    resparams: dictionary containing the properties for each layer
    """

    # declare parameters
    input_params = {
        "num_properties": num_properties,
        "reservoir_size": reservoir_size,
        # "input_scale":
    }
    reservoir_params = {
        "reservoir_size": reservoir_size
        # "degree": ,
        # "radius": ,
        # "recall":
    }
    output_params = {
        "num_classes": num_classes,
        "training_length": training_length,
        "reservoir_size": reservoir_size
    }

    resparams = {"input_params": input_params,
                 "reservoir_params": reservoir_params,
                 "output_params": output_params}
    return resparams


def format_classes(old_classes):
    """

    Parameters
    ----------
    old_classes: NumPy array
        contains the un-formatted classes for the data

    Returns
    -------
    new_classes:
        formatted classes for the data
    """

    # determine how many classes are present and how many input vectors there are
    num_classes = np.unique(old_classes).size
    num_inputs = old_classes.size

    # create all zeros matrix of appropriate size
    new_classes = np.zeros((num_inputs, num_classes))

    # iterate through classes and adjust properly
    for idx, class_value in enumerate(old_classes):
        new_classes[idx, class_value] = 1

    return new_classes


if __name__ == "__main__":
    # get the starting time
    start_time = time.time()

    # specify document name
    file_name = "seeds_dataset.txt"

    # read, format, and split the data
    training_data, test_data, training_classes, test_classes = get_data(file_name)

    # get the parameters
    training_length, num_properties = training_data.shape
    num_classes = training_classes.shape[1]  # np.unique(training_classes).size

    properties = {
        "reservoir_size": 100,
        "training_length": training_length,
        "num_properties": num_properties,
        "num_classes": num_classes
    }
    resparams = get_params(**properties)    # parameters for the reservoir layers

    # generate the reservoir computer
    classifier = rc(resparams)

    # train the classifier
    classifier.train_reservoir(training_data, training_classes)

    print("Prediction results for dataset: " + file_name)

    # test on training data
    output_training = classifier.rc_classification(training_data)

    print("\n" + "#" * 55)
    print("\nClassifier performance on training dataset\n")
    print(classification_report(training_classes, output_training))

    # test on test data
    output_test = classifier.rc_classification(test_data)

    print("#" * 55 + "\n")

    print("#" * 55)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(test_classes, output_test))
    print("#"*55 + "\n")

    # print running time
    print("Running time: " + str(time.time() - start_time) + "sec")

    total_data = np.vstack((training_data, test_data))
    total_classes = np.vstack((training_classes, test_classes))
    total_output = np.vstack((output_training, output_test))
    plt.figure()

    plt.subplot(121)
    mplot(num_classes, total_data, total_classes)
    plt.title('True classes', {"size": 12})

    plt.subplot(122)
    mplot(num_classes, total_data, total_output)
    plt.title('Classifier result', {"size": 12})
    # plt.show()

