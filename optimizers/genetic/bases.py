from abc import ABC, abstractmethod


class BaseCrossing(ABC):

    @abstractmethod
    def __call__(self, first_parent, second_parent):
        pass


class BaseFitness(ABC):

    @abstractmethod
    def __call__(self, anfis):
        pass


class BaseMutation(ABC):

    @abstractmethod
    def __call__(self, chromosome):
        pass


class BaseSelection(ABC):

    @abstractmethod
    def __call__(self, chromosomes, fitnesses, select_count):
        pass
