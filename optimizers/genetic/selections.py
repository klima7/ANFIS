from abc import ABC, abstractmethod


class BaseSelection(ABC):

    @abstractmethod
    def __call__(self, chromosomes, fitnesses, select_count):
        pass


class RouletteWheelSelection(BaseSelection):

    def __call__(self, chromosomes, fitnesses, select_count):
        pass


class RankSelection(BaseSelection):

    def __init__(self, count):
        self.count = count

    def __call__(self, chromosomes, fitnesses, select_count):
        pass
