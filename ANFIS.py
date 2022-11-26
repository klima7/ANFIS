# -*- coding: utf-8 -*-
from time import time

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from helps_and_enhancers import calculate_combinations
from optimizers import BaseOptimizer


class ANFIS:

    def __init__(self, inputs, training_data: np.ndarray, expected_labels: np.ndarray, t_norm, s_norm,
                 operator_init_value=0.5):
        self.input_list = inputs
        self.input_number = len(inputs)
        self.training_data = training_data
        self.expected_labels = expected_labels
        self.t_norm = t_norm
        self.s_norm = s_norm

        self.premises = []
        for i in range(self.input_number):
            self.premises.append(self.input_list[i].get())

        self.premises_combinations = np.array(calculate_combinations(self))[:, ::-1]
        self.operators_variations = list(product([self.t_norm, self.s_norm], repeat=self.input_number-1))

        mf_combinations_count = np.prod([inp.n_functions for inp in self.input_list])
        op_variations_count = len(self.operators_variations)
        nodes_number = mf_combinations_count * op_variations_count

        print(f'Combinations of membership functions: {mf_combinations_count}')
        print(f'Variations of operators: {op_variations_count}')
        print(f'Total rules count: {nodes_number}')

        self.tsk = np.random.random((nodes_number, self.input_number + 1))
        self.op = [operator_init_value] * nodes_number

    # ----------------------- aux -----------------------

    def set_premises_parameters(self, fv):
        fv = np.array(fv).reshape(np.shape(self.premises))
        self.premises = fv
        for i in range(self.input_number):
            self.input_list[i].set(*fv[i])

    def set_tsk_parameter(self, tsk):
        self.tsk = tsk.reshape(np.shape(self.tsk))

    def set_op_parameter(self, op):
        self.op = np.array(op).flatten()

    # -------------------- estimating --------------------

    def output_to_labels(self, y_pred):
        rounded = np.round(y_pred.flatten()).astype(int)
        r_shape = np.shape(rounded)
        return np.max((np.min((rounded, np.ones(r_shape)), axis=0), np.zeros(r_shape)), axis=0)  # clamp 0-1

    def estimate_labels(self):
        return self.anfis_estimate_labels(self.premises, self.op, self.tsk)

    def anfis_estimate_labels(self, fv, op, tsk) -> np.ndarray:
        data = self.training_data
        self.set_premises_parameters(fv)
        tsk = np.reshape(tsk, np.shape(self.tsk))
        memberships = [self.input_list[x].fuzzify(data[x]) for x in range(self.input_number)]

        # Wnioskowanie
        arguments = []
        for premises in self.premises_combinations:
            item = []
            for i in range(len(premises)):
                item.append(np.array(memberships[i])[:, premises[i]])
            arguments.append(item)

        arguments = np.transpose(arguments, (1, 2, 0))
        R = self.apply_all_operators_variations(arguments)

        # Normalizacja normalizacja poziomów aktywacji reguł
        Rsum = np.sum(R, axis=1, keepdims=True)

        Rnorm = np.divide(R, Rsum, out=np.zeros_like(R), where=Rsum != 0)
        Rnorm[(Rsum == 0).flatten(), :] = 0
        # wylicz wartoci przesłanek dla każdej próbki

        dataXYZ1 = np.vstack((self.training_data, np.ones(len(self.training_data[0])))).T
        Q = np.dot(dataXYZ1, tsk.T)

        # wyznacz wyniki wnioskowania dla każdej próbki
        result = (Q * Rnorm).sum(axis=1, keepdims=True)

        return result.T

    def apply_all_operators_variations(self, arguments):
        partial_results = []
        for operators_variation in self.operators_variations:
            partial_result = self.apply_single_operators_variation(arguments, operators_variation)
            partial_results.append(partial_result)
        return np.hstack(partial_results)

    @staticmethod
    def apply_single_operators_variation(arguments, operators_variation):
        result = operators_variation[0](np.array([arguments[0], arguments[1]]), None)
        for i in range(1, len(operators_variation)):
            result = operators_variation[i](np.array([result, arguments[i]+1]), None)
        return result

    # ----------------------- training -----------------------

    def set_training_and_testing_data(self, training_data, expected_labels):
        self.training_data = training_data
        self.expected_labels = expected_labels

    def train(self, optimiser: BaseOptimizer):
        start = time()
        optimiser.optimize(self)
        duration = time() - start
        print(f'Optimization finished after {duration:.2f}s')

    # -------------------- vizualization functions --------------------

    def show_results(self, color=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if color is None:
            color = [[1, 0, 0] if cc else [0, 1, 0] for cc in self.expected_labels]

        result = self.anfis_estimate_labels(self.premises, self.op, self.tsk)
        ax.scatter(self.training_data[0], self.training_data[1], result, c=color)
        plt.show()

    def show_inputs(self):
        plt.figure()
        for i in range(self.input_number):
            plt.subplot(self.input_number, 1, i + 1)
            self.input_list[i].show()
            plt.legend()
        plt.show()
