from enum import Enum
from posixpath import split
from turtle import right
import numpy as np

class FeatureType(Enum):
    CATEGORICAL = 0
    NUMERICAL = 1


def get_type(feature):
    numeric = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if feature.dtype in numeric:
        return FeatureType.NUMERICAL
    return FeatureType.CATEGORICAL


def numerical_split(X, col):
    min = X[col].min()
    max = X[col].max()
    split_points = np.linspace(min, max, 12)

    splits = []

    for split in split_points[1:11]:
        left = X.index[X[col] <= split] 
        right = X.index[X[col] > split]
        split_value = split
        splits.append((left, right, split_value))
    
    return splits


def categorical_split(X, col):
    split_points = X[col].unique()
    
    splits = []

    for split in split_points:
        left =  X.index[X[col] == split]
        right = X.index[X[col] != split]
        split_value = split
        splits.append((left, right, split_value))

    return splits