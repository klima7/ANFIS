from abc import ABC, abstractmethod


class BaseFitness(ABC):

    @abstractmethod
    def __call__(self, anfis):
        pass


class SmallestMaeErrorFitness(BaseFitness):

    def __call__(self, anfis):
        pass
