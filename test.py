
from params import FuzzyInputVariable_2Trapezoids
import numpy as np
from ANFIS import ANFIS
import time
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from optimizers import DefaultOptimizer, GeneticOptimizer
from optimizers.genetic.components import SmallestMaeErrorFitness, MultiPointCrossing, NRandomChangesMutation, RouletteWheelSelection, RankSelection


def generate_dataset():
    x = np.arange(0, 1, 0.1)
    x, y = np.meshgrid(x, x)

    dataX = x.flatten()
    dataY = y.flatten()
    dataXY = np.column_stack((dataX, dataY))
    data_labels = np.logical_xor(dataX >= 0.5, dataY >= 0.5)
    return train_test_split(dataXY, data_labels, test_size=0.2, random_state=25)


X_train, X_test, y_train, y_test = generate_dataset()

varX = FuzzyInputVariable_2Trapezoids(0.5, 0.5, "XAxis", ["L","H"]) # low, high
varY = FuzzyInputVariable_2Trapezoids(0.5, 0.5, "YAxis", ["L","H"])
fis = ANFIS([varX, varY], X_train.T, y_train)

default_opt = DefaultOptimizer(True, True, False, True, n_iter=50)

genetic_opt = GeneticOptimizer(
    SmallestMaeErrorFitness(),
    MultiPointCrossing(3),
    NRandomChangesMutation(2),
    RankSelection(5),
    cross_prob=0.7,
    mutate_prob=0.1,
    n_chromosomes=100,
    n_generations=5000,
    n_elite=2,
    learn_operators=False
)

optimizer = genetic_opt

fis.train(optimizer)

X_test = X_train
y_test = y_train

fis.training_data = X_test.T
y_pred = fis.estimate_labels()
y_pred = list(map(round, y_pred.flatten()))
print(confusion_matrix(y_test, y_pred))
