# coding=utf-8
from rcclassifier.Components import InputLayer, Reservoir, OutputLayer
import numpy as np


class ReservoirComputer:
    """
    Houses the actual reservoir computer

    ...

    Attributes
    ----------
    input_layer: InputLayer
    reservoir: Reservoir
    output_layer: Output

    Methods
    -------

    """

    def __init__(self, resparams):
        # get system parameter dictionaries
        input_params = resparams["input_params"]
        reservoir_params = resparams["reservoir_params"]
        output_params = resparams["output_params"]

        # construct the layers
        self.input_layer = InputLayer(**input_params)
        self.reservoir = Reservoir(**reservoir_params)
        self.output_layer = OutputLayer(**output_params)

    def train_reservoir(self, training_data, true_output):
        """
        Generates the output layer matrix based on the provided training data and true output

        Parameters
        ----------
        training_data: NumPy array
            size: (number of properties) x (number of entries)
        true_output: NumPy array
            size: (size of output) * (number of entries)

        Returns
        -------
        None
        """

        # declare the states matrix for keeping track of reservoir states
        states = np.zeros((self.output_layer.training_length, self.reservoir.reservoir_size))

        # iterate down the training data
        for idx, cur_input in enumerate(training_data):
            # update the next input for the reservoir computer
            self.input_layer.update_next_input(cur_input)

            # update the reservoir state
            next_state = self.reservoir.next_reservoir_state(self.input_layer)
            self.reservoir.update_reservoir_state(next_state)

            # store the result in the state matrix
            states[idx, :] = self.reservoir.current_state

        # generate the output layer matrix
        # found as the product of the pseudo-inverse of state matrix and the true_output matrix
        self.output_layer.output_matrix = np.matmul(np.linalg.pinv(states), true_output)

    def predict_single_class(self, input_vector):
        """
        Generates the reservoir state determined by input vector and decides the class

        Parameters
        ----------
        input_vector: NumPy array
            property vector that is to be classified

        Returns
        -------
        vector corresponding to the appropriate class
        """

        # update the input vector
        self.input_layer.update_next_input(input_vector)

        # step the reservoir
        next_state = self.reservoir.next_reservoir_state(self.input_layer)

        # determine result
        result = np.matmul(next_state, self.output_layer.output_matrix)

        return self.closest_class(result)

    def closest_class(self, vector):
        """

        Parameters
        ----------
        vector: NumPy array
            result of prediction

        Returns
        -------
        closest:
            the class with the closest Euclidean distance to the result vector
        """

        # get the number of classes
        num_classes = self.output_layer.num_classes

        # create class matrix (rows represent each of the possible classes)
        classes = np.eye(num_classes)

        # find which class is the closest
        closest = classes[0]
        for idx, c in enumerate(classes):
            if np.linalg.norm(vector - c) < np.linalg.norm(vector - closest):
                closest = c

        return closest

    def rc_classification(self, test_data):
        """

        Parameters
        ----------
        test_data: NumPy array
            data to be classified

        Returns
        -------
        result:
            result of the class predictions
            size: length of (test_data) x (num_classes)
        """
        # create matrix to hold output data
        test_length, _ = test_data.shape
        num_classes = self.output_layer.num_classes

        result = np.zeros((test_length, num_classes))

        for idx, next_input in enumerate(test_data):
            result[idx, :] = self.predict_single_class(next_input)

        return result

    @staticmethod
    def get_separated_by_class(data, classes, num_classes):
        """
        Takes data and classes and separates data into appropriate classes

        Returns
        -------
        organized_data
        """
        organized_data = np.zeros(data.shape + (num_classes, ))
        class_vectors = np.eye(num_classes)

        # go through each of the possible classes
        for idx, row in enumerate(class_vectors):
            cur_indices = np.where((classes == row).all(axis=1))
            cur_data = data[cur_indices]
            organized_data[:cur_data.shape[0], :, idx] = cur_data

        return organized_data

