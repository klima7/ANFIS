import numpy as np

from .bases import BaseSelection, BaseMutation, BaseCrossing, BaseFitness


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


class NRandomChangesMutation(BaseMutation):

    def __init__(self, n_changes):
        self.n_changes = n_changes

    def __call__(self, chromosome):
        indexes = np.arange(len(chromosome))
        np.random.shuffle(indexes)
        indexes = indexes[:self.n_changes]
        chromosome[indexes] = np.random.random((self.n_changes,))


class MultiPointCrossing(BaseCrossing):

    def __init__(self, n_points=1):
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


class SmallestMaeErrorFitness(BaseFitness):

    def __call__(self, anfis):
        predicted_labels = anfis.estimate_labels()
        error = np.sum(np.abs(predicted_labels - anfis.expected_labels))
        return 1 / error
