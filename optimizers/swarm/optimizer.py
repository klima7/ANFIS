import numpy as np

from .. import BaseOptimizer
from .solver import PSOSolver


class SwarmOptimizer(BaseOptimizer):

    def __init__(
            self,

            # swarm optimization params
            phases,
            n_particles=100,
            v=0.5,

            # stop conditions
            max_iters=1000,
            tol=1e-3,
            n_tol=0.7,

            # what should learn
            learn_premises=True,
            learn_operators=True,
            learn_consequents=True,

            # parameters bounds
            bounds_premises=(0, 4),
            bounds_operators=(0, 2),
            bounds_consequents=(0, 2),
    ):
        self.solver = PSOSolver(phases, n_particles, v, max_iters, tol, n_tol)

        self.bounds_premises = bounds_premises
        self.bounds_operators = bounds_operators
        self.bounds_consequents = bounds_consequents

        self.learn_premises = learn_premises
        self.learn_operators = learn_operators
        self.learn_consequents = learn_consequents

        # aux variables
        self.anfis = None
        self.premises_length = 0
        self.operators_length = 0
        self.consequents_length = 0

    def optimize(self, anfis):
        self._init_aux_variables(anfis)
        constraints = self._construct_domain_constraints()
        best_particle, _ = self.solver.solve(self._optimized_function, constraints)

    def _init_aux_variables(self, anfis):
        self.anfis = anfis
        self.premises_length = np.array(anfis.premises).size
        self.operators_length = len(anfis.op)
        self.consequents_length = len(anfis.tsk.flatten())

    def _construct_domain_constraints(self):
        premises_constraints = [self.bounds_premises] * self.premises_length if self.learn_premises else []
        operators_constraints = [self.bounds_operators] * self.operators_length if self.learn_operators else []
        consequents_constraints = [self.bounds_consequents] * self.consequents_length if self.learn_consequents else []

        return [*premises_constraints, *operators_constraints, *consequents_constraints]

    def _optimized_function(self, particle):
        weights = self._get_weights_from_particle(particle)
        self._config_anfis_from_weights(self.anfis, weights)
        predicted_labels = self.anfis.estimate_labels()
        error = np.sum(np.abs(predicted_labels - self.anfis.expected_labels))
        return error

    def _get_weights_from_particle(self, particle):
        all_weights = []
        current_pos = 0

        learn_flags = [self.learn_premises, self.learn_operators, self.learn_consequents]
        params_lengths = [self.premises_length, self.operators_length, self.consequents_length]

        for learn_flag, param_length in zip(learn_flags, params_lengths):
            if learn_flag:
                weights = particle[current_pos:current_pos + param_length]
                all_weights.append(weights)
                current_pos += param_length
            else:
                all_weights.append(None)

        return all_weights

    @staticmethod
    def _config_anfis_from_weights(anfis, weights):
        premises, operators, consequents = weights
        if premises is not None:
            anfis.set_premises_parameters(premises)
        if operators is not None:
            anfis.set_op_parameter(operators)
        if consequents is not None:
            anfis.set_tsk_parameter(consequents)
