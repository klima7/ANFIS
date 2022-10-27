from abc import ABC, abstractmethod


class BaseCrossing(ABC):

    @abstractmethod
    def __call__(self, first_parent, second_parent):
        pass


class MultiPointCrossing(BaseCrossing):

    def __init__(self, n_points):
        pass

    def __call__(self, first_parent, second_parent):
        pass
