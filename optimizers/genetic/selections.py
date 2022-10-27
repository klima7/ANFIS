import numpy as np

from abc import ABC, abstractmethod


class BaseSelection(ABC):

    @abstractmethod
    def __call__(self, chromosomes, fitnesses, select_count):
        pass


class RouletteWheelSelection(BaseSelection):

    def __call__(self, chromosomes, fitnesses, select_count):
        all_indexes = np.arange(len(chromosomes))
        selected_indexes = np.random.choice(all_indexes, size=select_count, p=fitnesses)
        return chromosomes[selected_indexes]


class RankSelection(BaseSelection):

    def __init__(self, count):
        self.count = count

    def __call__(self, chromosomes, fitnesses, select_count):
        pass
