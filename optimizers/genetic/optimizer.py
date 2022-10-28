from typing import Tuple

from einops import rearrange
import numpy as np
from tqdm import tqdm

from ..base import BaseOptimizer
from .bases import BaseFitness, BaseMutation, BaseCrossing, BaseSelection


class GeneticOptimizer(BaseOptimizer):

    def __init__(
            self,
            fitness: BaseFitness,
            crossing: BaseCrossing,
            mutation: BaseMutation,
            selection: BaseSelection,

            cross_prob: float = 0.7,
            mutate_prob: float = 0.1,

            n_chromosomes: int = 100,
            n_generations: int = 1000,
            n_elite: int = 0,

            bounds_premises: Tuple[float, float] = (0, 4),
            bounds_operators: Tuple[float, float] = (0, 2),
            bounds_consequents: Tuple[float, float] = (0, 2),

            learn_premises: bool = True,
            learn_operators: bool = True,
            learn_consequents: bool = True
    ):
        self.fitness = fitness
        self.crossing = crossing
        self.mutation = mutation
        self.selection = selection

        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob

        self.n_chromosomes = n_chromosomes
        self.n_generations = n_generations
        self.n_elite = n_elite

        self.bounds_premises = bounds_premises
        self.bounds_operators = bounds_operators
        self.bounds_consequents = bounds_consequents

        self.learn_premises = learn_premises
        self.learn_operators = learn_operators
        self.learn_consequents = learn_consequents

        self.genotypes = None
        self.fitnesses = None

        # aux
        self.anfis = None
        self.premises_length = 0
        self.operators_length = 0
        self.consequents_length = 0

    def optimize(self, anfis):
        self.anfis = anfis
        self._init_aux_variables(anfis)

        parameters_count = self._get_genotype_length()
        print(f'Optimizing {parameters_count} parameters')

        self.genotypes = self._generate_population()
        self.fitnesses = self._get_fitnesses(self.genotypes)
        self._order_by_fitness()

        for generation in tqdm(range(self.n_generations), total=self.n_generations, desc='Evolution'):
            self._evolve()

            if generation % 100 == 0:
                print(self.fitnesses[0])

        best_weights = self._get_weights_from_genotype(self.genotypes[0])
        self._config_anfis_from_weights(self.anfis, best_weights)

    def _evolve(self):
        elite = np.array(self.genotypes[:self.n_elite])
        selected = self._select(self.genotypes)
        crossed = self._cross(selected)
        mutated = self._mutate(crossed)

        self.genotypes = np.vstack([elite, mutated])
        self.fitnesses = self._get_fitnesses(self.genotypes)
        self._order_by_fitness()

    def _select(self, chromosomes):
        return self.selection(chromosomes, self.fitnesses, self.n_chromosomes - self.n_elite)

    def _cross(self, genotypes):
        children = []
        np.random.shuffle(genotypes)
        paired_genotypes = rearrange(genotypes, '(n p) g -> n p g', p=2)
        for parent1, parent2 in paired_genotypes:
            children.extend(self.crossing(parent1, parent2))
        return np.array(children)

    def _mutate(self, genotypes):
        mutated = [self.mutation(genotype) for genotype in genotypes]
        return np.array(mutated)

    def _order_by_fitness(self):
        sorted_indexes = np.argsort(self.fitnesses)[::-1]
        self.genotypes = self.genotypes[sorted_indexes]
        self.fitnesses = self.fitnesses[sorted_indexes]

    def _get_fitnesses(self, genotypes):
        fitnesses = []
        for genotype in genotypes:
            weights = self._get_weights_from_genotype(genotype)
            self._config_anfis_from_weights(self.anfis, weights)
            fitness = self.fitness(self.anfis)
            fitnesses.append(fitness)
        return np.array(fitnesses)

    def _init_aux_variables(self, anfis):
        self.premises_length = len([item for sublist in anfis.premises for item in sublist])
        self.operators_length = len(anfis.op)
        self.consequents_length = len(anfis.tsk.flatten())

    def _generate_population(self):
        chromosome_length = self._get_genotype_length()
        population = np.random.random((self.n_chromosomes, chromosome_length))
        return population

    def _get_genotype_length(self):
        length = 0
        if self.learn_premises:
            length += self.premises_length
        if self.learn_operators:
            length += self.operators_length
        if self.learn_consequents:
            length += self.consequents_length
        return length

    @staticmethod
    def _config_anfis_from_weights(anfis, weights):
        premises, operators, consequents = weights
        if premises is not None:
            anfis.set_premises_parameters(premises)
        if operators is not None:
            anfis.set_op_parameter(operators)
        if consequents is not None:
            anfis.set_tsk_parameter(consequents)

    def _get_weights_from_genotype(self, genotype):
        all_weights = []
        current_pos = 0

        learn_flags = [self.learn_premises, self.learn_operators, self.learn_consequents]
        params_lengths = [self.premises_length, self.operators_length, self.consequents_length]
        params_bounds = [self.bounds_premises, self.bounds_operators, self.bounds_consequents]

        for learn_flag, param_length, bounds in zip(learn_flags, params_lengths, params_bounds):
            if learn_flag:
                norm_weights = genotype[current_pos:current_pos + param_length]
                weights = norm_weights * (bounds[1] - bounds[0]) + bounds[0]
                all_weights.append(weights)
                current_pos += param_length
            else:
                all_weights.append(None)

        return all_weights
