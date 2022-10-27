import numpy as np
from scipy.optimize import minimize, basinhopping

from . import BaseOptimizer


class DefaultOptimizer(BaseOptimizer):

    def __init__(self, global_optimization: bool = True, learn_premises: bool = True, learn_operators: bool = False,
                 learn_consequents: bool = True, n_iter=100, bounds_premises=None):
        self.learn_premises = learn_premises
        self.learn_operators = learn_operators
        self.learn_consequents = learn_consequents
        self.global_optimization = global_optimization
        self.n_iter = n_iter
        self.bounds_premises = bounds_premises

    def optimize(self, anfis):

        x1 = [item for sublist in anfis.premises for item in sublist]
        x1 = np.array(x1).flatten()
        x2 = anfis.op
        x3 = anfis.tsk.flatten()

        if self.bounds_premises is None:
            bfv = [(0, 4)] * len(x1)
        else:
            bfv = self.bounds_premises
        bop = [(0.0, 2.0)] * len(x2)
        btsk = [(0, 2)] * len(x3)

        niter_success = 100

        if self.learn_premises and self.learn_operators and self.learn_consequents:
            x0 = np.hstack((x1, x2, x3))
            anfis.end_x1 = len(x1)
            anfis.end_x2 = len(x1) + len(x2)

            bounds = bfv + bop + btsk

            if self.global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds, "args": (self)}
                res = basinhopping(goal_premises_operators_consequents, x0, minimizer_kwargs=minimizer_kwargs,
                                   niter=self.n_iter, niter_success=niter_success)
            else:
                res = minimize(goal_premises_operators_consequents, x0, method='SLSQP', bounds=bounds, args=anfis)

            anfis.set_premises_parameters(res.x[:anfis.end_x1].reshape(np.shape(anfis.premises)))
            anfis.op = res.x[anfis.end_x1:anfis.end_x2]
            anfis.tsk = res.x[anfis.end_x2:].reshape(np.shape(anfis.tsk))

        elif self.learn_premises and self.learn_operators:
            x0 = np.hstack((x1, x2))
            anfis.end_x1 = len(x1)
            anfis.end_x2 = len(x1) + len(x2)

            bounds = bfv + bop

            if self.global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds}
                res = basinhopping(goal_premises_operators, x0, minimizer_kwargs=minimizer_kwargs, niter=self.n_iter,
                                   niter_success=niter_success)
            else:
                res = minimize(goal_premises_operators, x0, method='SLSQP', bounds=bounds, args=anfis)

            anfis.set_premises_parameters(res.x[:anfis.end_x1].reshape(np.shape(anfis.premises)))
            anfis.op = res.x[anfis.end_x1:anfis.end_x2]

        elif self.learn_premises and self.learn_consequents:
            x0 = np.hstack((x1, x3))
            anfis.end_x1 = len(x1)
            anfis.end_x2 = len(x1)

            bounds = bfv + btsk

            if self.global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds, "args": (anfis)}
                res = basinhopping(goal_premises_consequents, x0, minimizer_kwargs=minimizer_kwargs,
                                   niter=self.n_iter)  # , niter_success=niter_success)
            else:
                res = minimize(goal_premises_consequents, x0, method='SLSQP', bounds=bounds, args=anfis, tol=1e-6)

            anfis.set_premises_parameters(res.x[:anfis.end_x1])  ##zmiana funkcji
            anfis.tsk = res.x[anfis.end_x2:].reshape(np.shape(anfis.tsk))

        elif self.learn_operators and self.learn_consequents:
            x0 = np.hstack((x2, x3))
            anfis.end_x1 = 0
            anfis.end_x2 = len(x2)

            bounds = bop + btsk

            if self.global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds}
                res = basinhopping(goal_operators_consequents, x0, minimizer_kwargs=minimizer_kwargs, niter=self.n_iter,
                                   niter_success=niter_success)
            else:
                res = minimize(goal_operators_consequents, x0, method='SLSQP', bounds=bounds, args=anfis)

            anfis.op = res.x[anfis.end_x1:anfis.end_x2]
            anfis.tsk = res.x[anfis.end_x2:].reshape(np.shape(anfis.tsk))

        elif self.learn_premises:
            x0 = x1
            anfis.end_x1 = len(x1)
            anfis.end_x2 = len(x1)

            bounds = bfv

            if self.global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds}
                res = basinhopping(goal_premises, x0, minimizer_kwargs=minimizer_kwargs, niter=self.n_iter,
                                   niter_success=niter_success)
            else:
                res = minimize(goal_premises, x0, method='SLSQP', bounds=bounds, args=anfis)

            anfis.set_premises_parameters(res.x[:].reshape(np.shape(anfis.premises)))

        elif self.learn_operators:
            x0 = x2
            anfis.end_x1 = 0
            anfis.end_x2 = len(x2)

            bounds = bop

            if self.global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds}
                res = basinhopping(goal_operators, x0, minimizer_kwargs=minimizer_kwargs, niter=self.n_iter,
                                   niter_success=niter_success)
            else:
                res = minimize(goal_operators, x0, method='SLSQP', bounds=bounds, args=anfis)

            anfis.op = res.x[:]

        elif self.learn_consequents:
            x0 = x3
            anfis.end_x1 = 0
            anfis.end_x2 = 0

            bounds = btsk

            if self.global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds, "args": (anfis), "tol": 1e-03}
                res = basinhopping(goal_consequents, x0, minimizer_kwargs=minimizer_kwargs, niter=self.n_iter,
                                   niter_success=niter_success)
            else:
                res = minimize(goal_consequents, x0, method='SLSQP', bounds=bounds, args=anfis)

            anfis.tsk = res.x[:].reshape(np.shape(anfis.tsk))

        else:
            print("Error")
            assert (0)


def goal_premises_operators_consequents(input, self):
    fv = input[:self.end_x1].reshape(np.shape(self.premises))
    op = input[self.end_x1:self.end_x2]
    tsk = input[self.end_x2:]
    new_labels = self.anfis_estimate_labels(fv, op, tsk)

    error = (np.abs(new_labels - self.expected_labels)).sum()
    # error = (np.abs(new_labels - self.expected_labels)*self.entropy ).sum()
    # error = np.sqrt(np.abs(new_labels - dataC).sum())
    return error


def goal_premises_operators(input, self):
    fv = input[:self.end_x1].reshape(np.shape(self.premises))
    op = input[self.end_x1:self.end_x2]
    tsk = self.tsk
    new_labels = self.anfis_estimate_labels(fv, op, tsk)

    error = (np.abs(new_labels - self.expected_labels)).sum()
    return error


def goal_premises_consequents(input, self):
    fv = []
    last = 0
    for i in range(len(self.premises)):
        fv.append(input[last:last + len(self.premises[i])])
        last = len(fv)
    fv = np.reshape(input[:self.end_x2], np.shape(self.premises))  # np.array(fv)
    op = self.op
    tsk = input[self.end_x2:]
    new_labels = self.anfis_estimate_labels(fv, op, tsk)

    error = (np.abs(new_labels - self.expected_labels)).sum()
    return error


def goal_operators_consequents(input, self):
    fv = self.premises  # np.array(self.premises).flatten()
    op = input[self.end_x1:self.end_x2]
    tsk = input[self.end_x2:]
    new_labels = self.anfis_estimate_labels(fv, op, tsk)

    error = (np.abs(new_labels - self.expected_labels)).sum()
    return error


def goal_premises(input, self):
    fv = input[:self.end_x1].reshape(np.shape(self.premises))
    op = self.op
    tsk = self.tsk
    new_labels = self.anfis_estimate_labels(fv, op, tsk)

    error = (np.abs(new_labels - self.expected_labels)).sum()
    return error


def goal_operators(input, self):
    fv = self.premises  # np.array(self.premises).flatten()
    op = input[self.end_x1:self.end_x2]
    tsk = self.tsk
    new_labels = self.anfis_estimate_labels(fv, op, tsk)

    error = (np.abs(new_labels - self.expected_labels)).sum()
    return error


def goal_consequents(input, self):
    fv = self.premises
    op = self.op
    tsk = input[self.end_x2:]
    new_labels = self.anfis_estimate_labels(fv, op, tsk)

    error = (np.abs(new_labels - self.expected_labels)).sum()
    return error
