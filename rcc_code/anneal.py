import math
import random
import matplotlib.pyplot as plt
from rcclassifier.ReservoirComputer import ReservoirComputer as rc
from rcclassifier.myplotter import my_plotter as mplot
import numpy as np
import warnings

from run_classifier import format_classes, get_params


class SimAnneal(object):
    def __init__(self, data, properties, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.data = data
        self.num_classes = properties["num_classes"]
        self.classifier = rc(get_params(**properties))
        self.N = len(data)
        self.T = math.sqrt(self.N) if T == -1 else T
        self.T_save = self.T  # save initial T to reset if batch annealing is used
        self.alpha = 0.95 if alpha == -1 else alpha
        self.stopping_temperature = 1e-8 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

        self.nodes = [i for i in range(self.N)]
        self.cur_solution = None
        self.cur_fitness = float("Inf")
        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []

    def initial_solution(self):
        """
        Greedy algorithm to get an initial solution (closest-neighbour).
        """
        cur_classes = np.random.randint(0, self.num_classes, self.N)
        solution = format_classes(cur_classes.astype(int))

        cur_fit = self.fitness(solution)
        self.best_fitness = cur_fit
        self.best_solution = solution

        self.fitness_list.append(cur_fit)
        return solution, cur_fit

    def fitness(self, classes):
        """
        Mean standard deviation for the current solution
        """
        organized_sol = rc.get_separated_by_class(self.data, classes, self.num_classes)

        deviations = np.zeros((self.num_classes, 1))
        for i in range(self.num_classes):

            deviations[i] = SimAnneal.rms(organized_sol[:, :, i])

        return np.mean(deviations)

    def p_accept(self, candidate_fitness):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    def accept(self, candidate):
        """
        Accept with probability 1 if candidate is better than current.
        Accept with probabilty p_accept(..) if candidate is worse.
        """
        candidate_fitness = self.fitness(candidate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if candidate_fitness < self.cur_fitness:
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate
                if candidate_fitness < self.best_fitness and ~math.isnan(candidate_fitness):
                    self.best_fitness, self.best_solution = candidate_fitness, candidate
            else:
                # if fitness is greater, with a certain probability, scramble the classes
                if random.random() < self.p_accept(candidate_fitness) or math.isnan(candidate_fitness):
                    """
                    cur_classes = np.random.randint(0, self.num_classes, self.N)
                    self.cur_solution = format_classes(cur_classes.astype(int))

                    self.cur_fitness = self.fitness(self.cur_solution)
                    """
                    if random.random() < 0.5:
                        cur_classes = np.random.randint(0, self.num_classes, self.N)
                        self.cur_solution = format_classes(cur_classes.astype(int))
    
                        self.cur_fitness = self.fitness(self.cur_solution)
                    else:
                        self.cur_solution = self.best_solution
                        self.cur_fitness = self.best_fitness

    def anneal(self):
        """
        Execute simulated annealing algorithm.
        """
        # Initialize with the greedy solution.
        self.cur_solution, self.cur_fitness = self.initial_solution()
        plt.figure()
        self.mplot(self.data, self.cur_solution)
        plt.show()
        plt.close()
        print("Starting annealing.")
        plt.figure()
        # self.make_plot()
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            # produce a new classes candidate using false-reservoir computing
            self.classifier.train_reservoir(self.data, self.cur_solution)
            candidate = self.classifier.rc_classification(self.data)

            self.accept(candidate)
            self.make_plot()
            plt.show()
            self.T *= self.alpha
            self.iteration += 1

            self.fitness_list.append(self.cur_fitness)

        print("Best fitness obtained: ", self.best_fitness)

    def batch_anneal(self, times=10):
        """
        Execute simulated annealing algorithm `times` times, with random initial solutions.
        """
        for i in range(1, times + 1):
            print("Iteration {i}/{times} -------------------------------")
            self.T = self.T_save
            self.iteration = 1
            self.cur_solution, self.cur_fitness = self.initial_solution()
            self.anneal()

    def plot_this(self):
        self.mplot(self.data, self.cur_solution)

    def plot_best(self):
        self.mplot(self.data, self.best_solution)

    def make_plot(self):
        plt.close()
        plt.subplot(121)
        self.plot_best()
        plt.title('Best Solution', {"size": 8})

        plt.subplot(122)
        self.plot_this()
        plt.title('Current Solution', {"size": 8})
        plt.show()

    def mplot(self, data, classes):
        separated_data = rc.get_separated_by_class(data, classes, self.num_classes)

        for i in range(self.num_classes):
            cur_data = separated_data[:, :, i]
            cur_data = cur_data[~np.all(cur_data == 0, axis=1)]
            plt.plot(cur_data[:, 0], cur_data[:, 1], "o")

    @staticmethod
    def rms(data):
        data = data.transpose()
        cur_sum = 0
        for col in data:
            prop = col[col != 0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                cur_sum += np.mean(abs(prop - prop.mean())**2)
        return np.sqrt(cur_sum)
