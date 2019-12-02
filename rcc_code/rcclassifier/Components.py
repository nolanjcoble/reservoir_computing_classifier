# coding=utf-8

"""
Houses the components of a reservoir computer

Classes
-------
InputLayer :
    houses components of the input layer

Reservoir :
    houses components of the reservoir

OutputLayer :
    houses components of the output layer
"""

# import packages
import numpy as np
from scipy.sparse import random, linalg


class InputLayer:
    """
    Houses components of the input layer

    ...

    Attributes
    ----------
    num_properties: int
        number of properties that the input will have
    reservoir_size: int
        number of perceptrons in the reservoir
    input_scale: double
        parameter to scale the input vectors
    input_matrix: NumPy array
        matrix used to couple an input vector to the reservoir layer
    next_input: NumPy array
        vector containing the next input to the reservoir

    Methods
    -------
    generate_input_matrix(self)
        creates the input matrix for the reservoir computer
    update_next_input(self,next)
        feeds the next input data to the input layer
    """

    def __init__(self, num_properties, reservoir_size, input_scale=0.3):
        """

        Parameters
        ----------
        num_properties: int
            number of properties that the input will have
        reservoir_size: int
            number of perceptrons in the reservoir
        sigma: double
            parameter to scale the input vectors (default is 0.3)
        """

        self.num_properties = num_properties
        self.reservoir_size = reservoir_size
        self.input_scale = input_scale
        self.next_input = None

        # get the input matrix from the parameters
        self.input_matrix = self.generate_input_matrix()

    def generate_input_matrix(self):
        """Generates the input matrix for the reservoir computer

        Returns
        -------
        input_matrix: NumPy array
            matrix used to couple an input vector to the reservoir layer
        """

        # generate an initial matrix
        # uniformly distributed values from 0 to 1
        # size is num_properties x reservoir_size (for correct matrix multiplication dimensions)
        input_matrix = np.random.rand(self.num_properties, self.reservoir_size)

        # adjust to be uniformly distributed from -1 to 1
        input_matrix = 2 * input_matrix - 1

        # adjust by the input scaling, sigma, and return
        return self.input_scale * input_matrix

    def update_next_input(self, next_input):
        # adjust the next input
        self.next_input = next_input


class Reservoir:
    """
    Houses the components of the actual reservoir

    ...

    Attributes
    ----------
    reservoir_size: int
        number of perceptrons in the reservoir
    A: NumPy array
        adjacency matrix describing the perceptron connections
    degree: int
        average degree for the random A matrix
    radius: double
        spectral radius for the random A matrix
    recall: double
        value specifies how important the previous reservoir state is to the next
    current_state: NumPy array
        vector containing the current state of the reservoir

    Methods
    -------
    generate_reservoir_matrix(self)
        creates the random A matrix from the provided parameters
    """

    def __init__(self, reservoir_size, degree=3, radius=0.9, recall=0.0):
        """

        Parameters
        ----------
        reservoir_size: int
            number of perceptrons in the reservoir
        degree: int
            average degree for the random A matrix (default is 3)
        radius: double
            spectral radius for the random A matrix (default is 0.9)
        recall: double
            value specifies how important the previous reservoir state is to the next
        """

        self.reservoir_size = reservoir_size
        self.degree = degree
        self.radius = radius
        self.recall = recall

        # generate the perceptron connectivity matrix
        self.A = self.generate_reservoir_matrix()

        # initialize the reservoir to zeros
        self.current_state = np.zeros((1, reservoir_size))

    def generate_reservoir_matrix(self):
        """Generates the random connectivity matrix governing the reservoir

        Returns
        -------
        reservoir_matrix: NumPy array
        """

        # sparsity for the random matrix
        sparsity = self.degree / self.reservoir_size

        # generate the sparse random matrix
        reservoir_matrix = random(self.reservoir_size, self.reservoir_size, density=sparsity)

        # determine largest magnitude eigenvalue
        if self.reservoir_size < 8:
            val = 3
        else:
            val = 6

        eigenvalues, _ = linalg.eigs(reservoir_matrix, k=val)
        e = max(abs(eigenvalues))

        # adjust the spectral radius of the reservoir matrix and return
        return reservoir_matrix / e * self.radius

    def next_reservoir_state(self, input_layer):
        """

        Parameters
        ----------
        input_layer: InputLayer
            input layer being used to couple to the reservoir

        Returns
        -------
        next_state: the new state of the reservoir
        """
        next_state = self.recall * self.current_state + \
                             (1 - self.recall) * np.tanh(self.current_state @ self.A +
                                                         input_layer.next_input @ input_layer.input_matrix)
        return next_state

    def update_reservoir_state(self, next_state):
        # adjust the next state
        self.current_state = next_state


class OutputLayer:
    """
    Houses components of the output layer

    ...

    Attributes
    ----------
    num_classes: int
        number of classes that the input will have
    reservoir_size: int
        number of perceptrons in the reservoir
    training_length: int
        amount of available training data
    output_matrix: NumPy array
        output layer of the reservoir computer

    Methods
    -------
    None
    """

    def __init__(self, num_classes, training_length, reservoir_size):
        """

        Parameters
        ----------
        num_classes: int
            number of classes that that the input will have
        training_length:
            amount of available training data
        reservoir_size:
            number of perceptrons in the reservoir (default is 1000)
        """

        self.num_classes = num_classes
        self.training_length = training_length
        self.reservoir_size = reservoir_size
        self.output_matrix = None
