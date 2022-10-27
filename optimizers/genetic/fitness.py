import numpy as np

from abc import ABC, abstractmethod


class BaseFitness(ABC):

    @abstractmethod
    def __call__(self, anfis):
        pass


class SmallestMaeErrorFitness(BaseFitness):

    def __call__(self, anfis):
        predicted_labels = anfis.estimate_labels()
        error = np.sum(np.abs(predicted_labels - anfis.expected_labels))
        return 1 / error
