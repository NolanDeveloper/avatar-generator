import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

WEIGHTS = [1, 1]
MODE = '100k'

if MODE == '100k':
    pass
elif MODE == '10k':
    pass
else:
    pass

"""
def match_dist(point1, point2):
    length = len(point1)
    matches = sum([point1[i] == point2[i] for i in range(length)])
    return length - matches
"""


def match_dist(point1, point2):
    matches = sum([w * (x == y) for x, y, w in zip(point1, point2, WEIGHTS)])
    return len(point1) - matches


MALE_DATA = pd.read_csv('cartoonset100k_males.csv', index_col=0)
MALE_ESTIM = NearestNeighbors(metric=match_dist).fit(MALE_DATA)


FEMALE_DATA = pd.read_csv('cartoonset100k_females.csv', index_col=0)
FEMALE_ESTIM = NearestNeighbors(metric=match_dist).fit(FEMALE_DATA)

assert(sum(MALE_DATA.columns == FEMALE_DATA.columns) == len(MALE_DATA.columns))
COLS = MALE_DATA.columns
K = 30


def load_database():
    return


def find_best_match(database, desired_parameters, is_male=None):
    if is_male is None:
        is_male = bool(bp.random.randint(0, 2))
    point = np.zeros([1, len(COLS)])
    for i, col in enumerate(COLS):
        point[0, i] = desired_parameters[col]
    if is_male:
        matches = MALE_ESTIM.kneighbors(point, K)
    else:
        matches = FEMALE_ESTIM.kneighbors(point, K)
    distances = matches[0][0]
    best_dist = distances[0]
    indices = matches[1][0]
    best_indices = indices[distances == best_dist]
    choice = np.random.choice(best_indices)
    if is_male:
        choice = MALE_DATA.index[choice]
    else:
        choice = FEMALE_DATA.index[choice]
    return (Path('cartoonset100k/') / choice).__str__()