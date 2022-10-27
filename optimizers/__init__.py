from abc import ABC, abstractmethod


class BaseOptimizer(ABC):

    @abstractmethod
    def optimize(self, anfis):
        pass


from .default import DefaultOptimizer
