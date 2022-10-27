import numpy as np

from abc import ABC, abstractmethod


class BaseCrossing(ABC):

    @abstractmethod
    def __call__(self, first_parent, second_parent):
        pass


class MultiPointCrossing(BaseCrossing):

    def __init__(self, n_points):
        self.n_points = n_points

    def __call__(self, first_parent, second_parent):
        split_indexes = np.arange(1, len(first_parent)+1)
        np.random.shuffle(split_indexes)
        split_indexes = split_indexes[:self.n_points]

        merged_parents = np.row_stack([first_parent, second_parent])
        parts = np.split(merged_parents, split_indexes, axis=1)

        for part in parts:
            np.random.shuffle(part)

        reordered_merged = np.hstack(parts)
        first_child, second_child = reordered_merged[0], reordered_merged[1]
        return first_child, second_child
