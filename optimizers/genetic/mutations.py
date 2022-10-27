from abc import ABC, abstractmethod


class BaseMutation(ABC):

    @abstractmethod
    def __call__(self, chromosome):
        pass


class RandomChangesMutation(BaseMutation):

    def __init__(self, n_changes):
        self.n_changes = n_changes

    def __call__(self, chromosome):
        pass
