import numpy as np

from abc import ABC, abstractmethod


class BaseMutation(ABC):

    @abstractmethod
    def __call__(self, chromosome):
        pass


class NRandomChangesMutation(BaseMutation):

    def __init__(self, n_changes):
        self.n_changes = n_changes

    def __call__(self, chromosome):
        indexes = np.arange(len(chromosome))
        np.random.shuffle(indexes)
        indexes = indexes[:self.n_changes]
        chromosome[indexes] = np.random.random((self.n_changes,))
